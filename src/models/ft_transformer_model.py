"""
FT-Transformer com Atenção Inter-Instâncias e Aproximação CUR

Implementação baseada em Gorishniy et al. (2021) — "Revisiting Deep Learning Models
for Tabular Data" — com extensão de atenção inter-instâncias (inspirada no SAINT)
e aproximação CUR para compressão da matriz de atenção.

Fluxo:
  X ∈ R^{n×d}
    → FeatureTokenizer → [n, d+1, D] (D tokens por instância, incluindo CLS)
    → FTTransformerBlocks (atenção entre features)
    → CLS embedding: [n, D]
    → InterInstanceAttentionCUR (atenção entre instâncias, com CUR opcional)
    → MLP head → logits

Analogia com LSSVM:
  LSSVM:         K_{ij} = kernel(x_i, x_j) ∈ R^{n×n}  → Nyström → classificação
  FT-Transformer: A_{ij} = softmax(q_i · k_j / √D) ∈ R^{n×n} → CUR   → classificação

Landmarks selecionados de X_train (espaço original), índices fixos usados em ambos os paradigmas.

Autor: Paulo Ricardo Bernardo Silva
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Blocos do FT-Transformer (atenção entre features)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converte features tabulares numéricas em sequência de tokens.

    Cada feature j gera um token de dimensão d_model:
        token_j = x_j * W_j + b_j,   W_j, b_j ∈ R^{d_model}

    Referência: Gorishniy et al. (2021), Seção 3.1.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_features]
        # output: [batch, n_features, d_model]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTBlock(nn.Module):
    """
    Bloco Transformer para atenção entre features (tokens de uma mesma instância).

    Pre-LN: LayerNorm → MultiheadAttention → residual → LayerNorm → FFN → residual
    """

    def __init__(self, d_model: int, n_heads: int, ffn_factor: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        d_ffn = d_model * ffn_factor
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, n_tokens, d_model]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h


# ─────────────────────────────────────────────────────────────────────────────
# Atenção inter-instâncias com aproximação CUR opcional
# ─────────────────────────────────────────────────────────────────────────────

def _truncated_pinv(W: torch.Tensor, tau_ratio: float = 0.1) -> torch.Tensor:
    """
    Pseudo-inversa truncada via SVD.

    Descarta valores singulares < tau_ratio * σ_max para estabilidade numérica
    (mesmo procedimento usado nos experimentos CUR do DistilBERT).

    Parâmetros
    ----------
    W : tensor [m, m]
    tau_ratio : float
        Threshold = tau_ratio × σ_max

    Retorna
    -------
    W_pinv : tensor [m, m]
    """
    # Verificação de segurança: se W tiver NaN/inf (modelo em divergência),
    # retornar zeros — a CUR não contribui para o output nesse passo.
    if not torch.isfinite(W).all():
        return torch.zeros(W.shape[0], W.shape[0], device=W.device, dtype=W.dtype)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    threshold = tau_ratio * S[0].abs().clamp(min=1e-10)
    S_inv = torch.where(S.abs() > threshold, 1.0 / S.clamp(min=1e-10),
                        torch.zeros_like(S))
    return Vh.mH @ torch.diag(S_inv) @ U.mH


class InterInstanceAttentionCUR(nn.Module):
    """
    Atenção inter-instâncias com aproximação CUR opcional.

    Dado um conjunto de n embeddings (um por instância), calcula:
        A_{ij} = softmax_j( q_i · k_j / √d_head )   ∈ R^{n×n}

    Com aproximação CUR (m landmarks):
        Ã = C @ U^{-1} @ R,   C = A[:, col_idx], R = A[row_idx, :], U = A[row_idx][:, col_idx]

    Os índices de landmark são pré-selecionados de X_train (espaço original) e
    passados como argumento — analogia direta com o LSSVM onde landmarks são
    selecionados de X_train antes da resolução do sistema.
    """

    def __init__(self, d_model: int, n_heads: int, tau_ratio: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.tau_ratio = tau_ratio
        self.scale = self.d_head ** 0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor,
                landmark_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parâmetros
        ----------
        h : tensor [n, d_model]
            Embeddings CLS de todas as instâncias no batch
        landmark_idx : tensor [m] ou None
            Índices dos landmarks. Se None, usa atenção completa.

        Retorna
        -------
        out : tensor [n, d_model]
        """
        n = h.shape[0]
        residual = h

        # [H, n, dh] — layout batchizado para operações BLAS eficientes
        Q = self.q_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)
        K = self.k_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)
        V = self.v_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)

        # S[H, n, n], A[H, n, n] — batched sobre todas as cabeças de uma vez
        S = Q @ K.transpose(-1, -2) / self.scale   # [H, n, n]
        A = torch.softmax(S, dim=-1)               # [H, n, n]

        if landmark_idx is None:
            # Atenção completa: path totalmente batchizado (sem loop Python)
            out = A @ V                            # [H, n, dh]
            out = out.permute(1, 0, 2).reshape(n, self.d_model)  # [n, d_model]
        else:
            # ── CUR: C = A[:, idx], R = A[idx, :], W = R[:, :, idx] ──
            # U^{-1} é computado em modo detached para evitar gradientes
            # instáveis através do SVD quando W tem valores singulares
            # degenerados (problema conhecido com torch.linalg.svd).
            # Gradientes fluem através de C e R (que dependem de A).
            idx = landmark_idx
            C = A[:, :, idx]                       # [H, n, m]
            R = A[:, idx, :]                       # [H, m, n]
            W = R[:, :, idx]                       # [H, m, m]

            # _truncated_pinv: loop por cabeça (SVD por cabeça)
            U_inv_list = []
            with torch.no_grad():
                for hd in range(self.n_heads):
                    U_inv_list.append(_truncated_pinv(W[hd], self.tau_ratio))
            U_inv = torch.stack(U_inv_list, dim=0)     # [H, m, m]

            A_approx = C @ U_inv @ R               # [H, n, n]
            out = A_approx @ V                     # [H, n, dh]
            out = out.permute(1, 0, 2).reshape(n, self.d_model)  # [n, d_model]

        out = self.out_proj(out)
        return self.norm(out + residual)


# ─────────────────────────────────────────────────────────────────────────────
# Opção 1: Nyströmformer-style — nunca materializa A completa
# Custo: O(nmd) vs O(n²d) da versão original
# ─────────────────────────────────────────────────────────────────────────────

class InterInstanceAttentionNystrom(nn.Module):
    """
    Atenção inter-instâncias com aproximação Nyström (computacionalmente honesta).

    Não materializa a matriz A ∈ R^{n×n}. Computa C e R com softmax separado
    sobre os m landmarks, depois aplica right-to-left:
        out = C @ (pinv(W) @ (R @ V))   — O(nmd_h) total

    C = softmax(Q @ K_m^T / √d)   [H, n, m]  — cada linha normalizada sobre m keys
    R = softmax(Q_m @ K^T  / √d)  [H, m, n]  — cada linha normalizada sobre n keys
    W = softmax(Q_m @ K_m^T / √d) [H, m, m]  — quadrado landmark × landmark
    """

    def __init__(self, d_model: int, n_heads: int, tau_ratio: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.tau_ratio = tau_ratio
        self.scale    = self.d_head ** 0.5

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor,
                landmark_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = h.shape[0]
        residual = h

        Q = self.q_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)  # [H,n,dh]
        K = self.k_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)
        V = self.v_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)

        if landmark_idx is None:
            # Atenção completa (fallback)
            A = torch.softmax(Q @ K.transpose(-1, -2) / self.scale, dim=-1)
            out = A @ V
        else:
            idx  = landmark_idx
            K_m  = K[:, idx, :]   # [H, m, dh]
            Q_m  = Q[:, idx, :]   # [H, m, dh]

            # C: cada linha normaliza sobre m landmark-keys  → [H, n, m]
            C = torch.softmax(Q @ K_m.transpose(-1, -2) / self.scale, dim=-1)
            # R: cada linha normaliza sobre todos n keys    → [H, m, n]
            R = torch.softmax(Q_m @ K.transpose(-1, -2) / self.scale, dim=-1)
            # W: landmark × landmark                        → [H, m, m]
            W = torch.softmax(Q_m @ K_m.transpose(-1, -2) / self.scale, dim=-1)

            U_inv_list = []
            with torch.no_grad():
                for hd in range(self.n_heads):
                    U_inv_list.append(_truncated_pinv(W[hd], self.tau_ratio))
            U_inv = torch.stack(U_inv_list, dim=0)   # [H, m, m]

            # Right-to-left para nunca materializar n×n
            RV   = R @ V                  # [H, m, dh]
            URV  = U_inv @ RV             # [H, m, dh]
            out  = C @ URV                # [H, n, dh]

        out = out.permute(1, 0, 2).reshape(n, self.d_model)
        out = self.out_proj(out)
        return self.norm(out + residual)


# ─────────────────────────────────────────────────────────────────────────────
# Opção 3: Linear attention + CUR (sem softmax → sem barreira O(n²))
# φ(x) = ELU(x) + 1  (sempre positivo, aproxima comportamento do softmax)
# Custo: O(nmd_h) — genuinamente sub-quadrático
# ─────────────────────────────────────────────────────────────────────────────

def _elu1(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x) + 1.0


class InterInstanceAttentionLinearCUR(nn.Module):
    """
    Atenção inter-instâncias com kernel linear (ELU+1) e aproximação CUR.

    φ(Q) e φ(K) substituem os vetores Q e K originais; a "matriz de atenção"
    linear L = φ(Q) @ φ(K)^T nunca é materializada. CUR é aplicada sobre L:

        C = φ(Q) @ φ(K_m)^T   [H, n, m]
        R = φ(Q_m) @ φ(K)^T   [H, m, n]
        W = C[:, idx, :]       [H, m, m]
        out = C @ (pinv(W) @ (R @ V))   — O(nmd_h)
    """

    def __init__(self, d_model: int, n_heads: int, tau_ratio: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.tau_ratio = tau_ratio
        self.scale     = self.d_head ** 0.5

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor,
                landmark_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = h.shape[0]
        residual = h

        Qr = _elu1(self.q_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2))  # [H,n,dh]
        Kr = _elu1(self.k_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2))
        V  = self.v_proj(h).view(n, self.n_heads, self.d_head).permute(1, 0, 2)

        if landmark_idx is None:
            # Linear attention completa (sem CUR): O(nd²) via associatividade
            KV  = Kr.transpose(-1, -2) @ V          # [H, dh, dh]
            out = Qr @ KV                            # [H, n, dh]
            # normalização por densidade (estabilidade)
            denom = (Qr @ Kr.transpose(-1, -2).sum(dim=-1, keepdim=True)).clamp(min=1e-6)
            out = out / denom
        else:
            idx  = landmark_idx
            Kr_m = Kr[:, idx, :]   # [H, m, dh]
            Qr_m = Qr[:, idx, :]   # [H, m, dh]

            C = Qr @ Kr_m.transpose(-1, -2)   # [H, n, m]
            R = Qr_m @ Kr.transpose(-1, -2)   # [H, m, n]
            W = C[:, idx, :]                   # [H, m, m]

            U_inv_list = []
            with torch.no_grad():
                for hd in range(self.n_heads):
                    U_inv_list.append(_truncated_pinv(W[hd], self.tau_ratio))
            U_inv = torch.stack(U_inv_list, dim=0)

            RV  = R @ V        # [H, m, dh]
            URV = U_inv @ RV   # [H, m, dh]
            out = C @ URV      # [H, n, dh]

        out = out.permute(1, 0, 2).reshape(n, self.d_model)
        out = self.out_proj(out)
        return self.norm(out + residual)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo completo: FT-Transformer Classificador
# ─────────────────────────────────────────────────────────────────────────────

class FTTransformerClassifier(nn.Module):
    """
    FT-Transformer para classificação binária de dados tabulares.

    Três modos:
      1. Baseline        : apenas atenção entre features (sem inter-instâncias)
      2. Inter-instâncias completa: atenção n×n entre instâncias (sem CUR)
      3. Inter-instâncias CUR     : atenção comprimida com m landmarks

    O modo é determinado em tempo de execução pelo argumento `landmark_idx`
    passado ao forward():
      - landmark_idx=None  → baseline ou full (controlado por use_inter_instance)
      - landmark_idx=tensor → CUR
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, use_inter_instance: bool = True,
                 tau_ratio: float = 0.1, dropout: float = 0.0,
                 attn_mode: str = "cur_full"):
        """
        attn_mode:
          "cur_full"    — original: materializa A completa (O(n²d)), CUR como pós-processamento
          "nystrom"     — opção 1: Nyströmformer, nunca materializa A (O(nmd))
          "linear_cur"  — opção 3: kernel ELU+1 + CUR (O(nmd), sem softmax)
        """
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model
        self.use_inter_instance = use_inter_instance
        self.attn_mode = attn_mode

        # Tokenização de features
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # Token CLS aprendível
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Blocos de atenção entre features
        self.blocks = nn.ModuleList([
            FTBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm_cls = nn.LayerNorm(d_model)

        # Atenção inter-instâncias (opcional)
        if use_inter_instance:
            _ATTN = {
                "cur_full":   InterInstanceAttentionCUR,
                "nystrom":    InterInstanceAttentionNystrom,
                "linear_cur": InterInstanceAttentionLinearCUR,
            }
            cls = _ATTN.get(attn_mode, InterInstanceAttentionCUR)
            self.inter_attn = cls(d_model, n_heads, tau_ratio=tau_ratio,
                                  dropout=dropout)

        # Cabeça de classificação
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def get_cls_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computa embeddings CLS via atenção entre features.

        Parâmetros
        ----------
        x : tensor [n, d_features]

        Retorna
        -------
        cls_embed : tensor [n, d_model]
        """
        n = x.shape[0]
        tokens = self.tokenizer(x)                      # [n, d, D]
        cls = self.cls_token.expand(n, -1, -1)          # [n, 1, D]
        tokens = torch.cat([cls, tokens], dim=1)         # [n, d+1, D]

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm_cls(tokens[:, 0, :])            # [n, D] — token CLS

    def forward(self, x: torch.Tensor,
                landmark_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parâmetros
        ----------
        x : tensor [n, d_features]
        landmark_idx : tensor [m] ou None
            Índices dos landmarks para CUR. None = sem CUR (atenção completa ou baseline).

        Retorna
        -------
        logits : tensor [n]
        """
        cls_embed = self.get_cls_embeddings(x)          # [n, D]

        if self.use_inter_instance:
            cls_embed = self.inter_attn(cls_embed, landmark_idx)

        logits = self.head(cls_embed).squeeze(-1)        # [n]
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# SAINT — Self-Attention and Intersample Attention Transformer
# Somepalli et al. (2021): "SAINT: Improved Neural Networks for Tabular Data"
# ─────────────────────────────────────────────────────────────────────────────

class SAINTBlock(nn.Module):
    """
    Bloco SAINT: alterna feature-attention (sobre tokens) e intersample attention
    (sobre instâncias) em cada camada.

    Diferença-chave vs. FTTransformerClassifier (inter_full):
      - inter_full: todos os blocos de feature-attention → uma camada inter-instâncias
      - SAINTBlock: feature-attention → inter-instâncias → feature-attention → ... (alternado)

    A intersample attention opera sobre o token CLS de cada instância, que
    agrega a representação intra-instância. Sem CUR (referência completa).
    """

    def __init__(self, d_model: int, n_heads: int, tau_ratio: float = 0.1,
                 dropout: float = 0.0):
        super().__init__()
        self.feat_block = FTBlock(d_model, n_heads, dropout=dropout)
        self.inter_attn = InterInstanceAttentionCUR(d_model, n_heads,
                                                     tau_ratio=tau_ratio,
                                                     dropout=dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros
        ----------
        h : tensor [n, n_tokens, d_model]  (CLS na posição 0)

        Retorna
        -------
        h : tensor [n, n_tokens, d_model]
        """
        h = self.feat_block(h)                  # feature-attention (por instância)
        cls = h[:, 0, :]                        # CLS tokens: [n, d_model]
        cls = self.inter_attn(cls, None)        # intersample attention completa (sem CUR)
        h = torch.cat([cls.unsqueeze(1), h[:, 1:, :]], dim=1)
        return h


class SAINTClassifier(nn.Module):
    """
    SAINT para classificação binária de dados tabulares.

    Usa blocos SAINTBlock (alternância feat-attn + inter-attn) como referência
    com atenção intersample completa (sem CUR). Serve de upper bound para
    a aproximação CUR proposta em FTTransformerClassifier.

    Reutiliza FeatureTokenizer e InterInstanceAttentionCUR existentes;
    compatível com fit_model() e eval_with_context() (landmark_idx=None).
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, tau_ratio: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([
            SAINTBlock(d_model, n_heads, tau_ratio=tau_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_cls = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                landmark_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parâmetros
        ----------
        x : tensor [n, d_features]
        landmark_idx : ignorado (SAINT usa atenção completa); aceito para compatibilidade
                       com fit_model() e eval_with_context().

        Retorna
        -------
        logits : tensor [n]
        """
        n = x.shape[0]
        h = self.tokenizer(x)                           # [n, d, D]
        cls = self.cls_token.expand(n, -1, -1)          # [n, 1, D]
        h = torch.cat([cls, h], dim=1)                  # [n, d+1, D]

        for block in self.blocks:
            h = block(h)

        cls_embed = self.norm_cls(h[:, 0, :])           # [n, D]
        return self.head(cls_embed).squeeze(-1)          # [n]


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares de treinamento
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model: FTTransformerClassifier, X_train: torch.Tensor,
                y_train: torch.Tensor, optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                landmark_idx: Optional[torch.Tensor] = None,
                max_grad_norm: float = 1.0) -> float:
    """Um passo de gradient descent com o batch completo."""
    model.train()
    optimizer.zero_grad()
    logits = model(X_train, landmark_idx)
    loss = criterion(logits, y_train)
    if torch.isfinite(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return loss.item() if torch.isfinite(loss) else float('nan')


@torch.no_grad()
def eval_with_context(model: FTTransformerClassifier,
                      X_ctx: torch.Tensor,
                      y_target: torch.Tensor,
                      n_context: int,
                      landmark_idx: Optional[torch.Tensor] = None) -> float:
    """
    Avalia instâncias de X_ctx[n_context:] usando X_ctx[:n_context] como contexto.

    Landmark indices referenciam posições dentro de X_ctx[:n_context], que têm
    índices 0..n_context-1 — sempre válidos independentemente do tamanho do conjunto
    de avaliação.

    Para o baseline (sem atenção inter-instâncias), avalia apenas X_ctx[n_context:]
    sem contexto extra.
    """
    model.eval()
    uses_inter = getattr(model, 'use_inter_instance', True)
    if not uses_inter:
        logits = model(X_ctx[n_context:], landmark_idx=None)
    else:
        logits_all = model(X_ctx, landmark_idx=landmark_idx)
        logits = logits_all[n_context:]
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == y_target).float().mean().item()


def fit_model(model: FTTransformerClassifier,
              X_train_t: torch.Tensor, y_train_t: torch.Tensor,
              X_val_t: torch.Tensor, y_val_t: torch.Tensor,
              landmark_idx: Optional[torch.Tensor] = None,
              lr: float = 1e-3, epochs: int = 300, patience: int = 30,
              weight_decay: float = 1e-4,
              early_stop_metric: str = "val_acc") -> dict:
    """
    Treina o modelo com early stopping em uma métrica de validação.

    A validação usa X_train como contexto (mesmo design da inferência final):
    concatena [X_train || X_val] e extrai predições para X_val.
    Landmark indices sempre referenciam X_train (posições 0..n_train-1).

    Parameters
    ----------
    early_stop_metric : str
        Métrica para early stopping:
          - "val_acc"      (default, comportamento original): maximiza acurácia
          - "val_loss"     : minimiza BCE loss
          - "val_f1_macro" : maximiza F1-macro (recomendado para imbalanceados)

    Retorna
    -------
    info : dict
        best_val_{metric}, n_epochs, train_time_s
    """
    import time
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    n_train = X_train_t.shape[0]
    X_ctx_val = torch.cat([X_train_t, X_val_t], dim=0)

    if early_stop_metric not in {"val_acc", "val_loss", "val_f1_macro"}:
        raise ValueError(f"early_stop_metric inválida: {early_stop_metric}")

    # Inicialização (val_acc/val_f1 maximizam; val_loss minimiza)
    best_score = float("inf") if early_stop_metric == "val_loss" else -float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(epochs):
        train_epoch(model, X_train_t, y_train_t, optimizer, criterion,
                    landmark_idx)

        # Computa logits/preds em val, espelhando exatamente a inferência final.
        # Para modelos com atenção inter-instâncias (SAINT, FT-CUR):
        # passamos [X_train || X_val] e extraímos logits[n_train:].
        # Para FT-Transformer baselines (sem inter-instâncias): só X_val.
        #
        # Bug capturado em code review (29/05/2026): a versão anterior
        # usava `landmark_idx is None` para decidir, o que quebrava SAINT
        # (que tem inter-instâncias mas sem landmarks) — validava sem
        # contexto enquanto a inferência final tinha contexto. Agora
        # usamos `model.use_inter_instance` (mesma lógica do
        # eval_with_context), simétrica entre SAINT e FT-CUR.
        model.eval()
        uses_inter = getattr(model, 'use_inter_instance', True)
        with torch.no_grad():
            if uses_inter:
                logits_all = model(X_ctx_val, landmark_idx=landmark_idx)
                logits = logits_all[n_train:]
            else:
                logits = model(X_val_t, landmark_idx=None)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            if early_stop_metric == "val_acc":
                score = (preds == y_val_t).float().mean().item()
                better = score > best_score
            elif early_stop_metric == "val_loss":
                score = criterion(logits, y_val_t).item()
                better = score < best_score
            else:  # val_f1_macro
                tp = ((preds == 1) & (y_val_t == 1)).sum().item()
                fp = ((preds == 1) & (y_val_t == 0)).sum().item()
                fn = ((preds == 0) & (y_val_t == 1)).sum().item()
                tn = ((preds == 0) & (y_val_t == 0)).sum().item()
                # F1 da classe positiva
                f1_pos = 2 * tp / max(2 * tp + fp + fn, 1)
                # F1 da classe negativa (positivos invertidos)
                f1_neg = 2 * tn / max(2 * tn + fn + fp, 1)
                score = (f1_pos + f1_neg) / 2
                better = score > best_score

        if better:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        f'best_{early_stop_metric}': best_score,
        'early_stop_metric': early_stop_metric,
        'n_epochs': epoch + 1,
        'train_time_s': time.time() - t0,
    }


def compute_approx_error(model: FTTransformerClassifier,
                         X_t: torch.Tensor,
                         landmark_idx: torch.Tensor,
                         n_heads_sample: int = 1) -> float:
    """
    Calcula o erro de reconstrução da atenção inter-instâncias (primeira cabeça).

    ||A - Ã||_F / ||A||_F   onde A é atenção completa e Ã é aproximação CUR.
    """
    model.eval()
    with torch.no_grad():
        cls_embed = model.get_cls_embeddings(X_t)          # [n, D]
        ia = model.inter_attn
        Q = ia.q_proj(cls_embed).view(cls_embed.shape[0], ia.n_heads, ia.d_head)
        K = ia.k_proj(cls_embed).view(cls_embed.shape[0], ia.n_heads, ia.d_head)

        # Usar apenas primeira cabeça para estimar erro (custo computacional)
        Qh, Kh = Q[:, 0, :], K[:, 0, :]
        S = Qh @ Kh.T / ia.scale
        A = torch.softmax(S, dim=-1)

        idx = landmark_idx
        C = A[:, idx]
        R = A[idx, :]
        W = A[idx][:, idx]
        U_inv = _truncated_pinv(W, ia.tau_ratio)
        A_approx = C @ U_inv @ R

        diff = A - A_approx
        err = (diff.norm(p='fro') / A.norm(p='fro')).item()
    return err
