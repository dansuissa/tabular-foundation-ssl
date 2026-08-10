"""Method capability registry for dual-view dispatch and resource routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.exceptions import UnsupportedMethodError

InputView = Literal["processed", "raw", "both"]
ProtocolKind = Literal["inductive", "transductive"]
DeviceKind = Literal["cpu", "gpu", "any"]
FidelityKind = Literal[
    "official",
    "faithful_reimplementation",
    "paper_core",
    "novel_experimental",
]
EnvKind = Literal["ssl-core", "ssl-tfm", "ssl-representation"]


@dataclass(frozen=True)
class MethodCapabilities:
    name: str
    family: str
    input_view: InputView
    uses_unlabeled_data: bool
    protocol: ProtocolKind
    device: DeviceKind
    supports_binary: bool
    supports_multiclass: bool
    fidelity: FidelityKind
    reference_paper: str | None
    upstream_commit: str | None
    needs_external_validation: bool
    env: EnvKind
    supports_predict_proba: bool = True


def _cap(
    name: str,
    *,
    family: str,
    input_view: InputView,
    uses_unlabeled_data: bool,
    protocol: ProtocolKind = "inductive",
    device: DeviceKind = "cpu",
    supports_binary: bool = True,
    supports_multiclass: bool = True,
    fidelity: FidelityKind,
    reference_paper: str | None = None,
    upstream_commit: str | None = None,
    needs_external_validation: bool = False,
    env: EnvKind = "ssl-core",
    supports_predict_proba: bool = True,
) -> MethodCapabilities:
    return MethodCapabilities(
        name=name,
        family=family,
        input_view=input_view,
        uses_unlabeled_data=uses_unlabeled_data,
        protocol=protocol,
        device=device,
        supports_binary=supports_binary,
        supports_multiclass=supports_multiclass,
        fidelity=fidelity,
        reference_paper=reference_paper,
        upstream_commit=upstream_commit,
        needs_external_validation=needs_external_validation,
        env=env,
        supports_predict_proba=supports_predict_proba,
    )


METHOD_CAPABILITIES: dict[str, MethodCapabilities] = {
    # ------------------------------------------------------------------
    # Legacy supervised (processed)
    # ------------------------------------------------------------------
    "logistic_regression": _cap(
        "logistic_regression",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="paper_core",
        reference_paper="sklearn LogisticRegression",
    ),
    "random_forest": _cap(
        "random_forest",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="paper_core",
        reference_paper="Breiman 2001 Random Forests",
    ),
    "xgboost": _cap(
        "xgboost",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="official",
        reference_paper="Chen & Guestrin 2016 XGBoost",
    ),
    "lightgbm": _cap(
        "lightgbm",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="official",
        reference_paper="Ke et al. 2017 LightGBM",
    ),
    "catboost": _cap(
        "catboost",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="official",
        reference_paper="Prokhorenkova et al. 2018 CatBoost",
    ),
    "mlp": _cap(
        "mlp",
        family="supervised",
        input_view="processed",
        uses_unlabeled_data=False,
        fidelity="paper_core",
        reference_paper="sklearn MLPClassifier",
    ),
    # ------------------------------------------------------------------
    # Legacy graph / self-training / RPL
    # ------------------------------------------------------------------
    "label_spreading": _cap(
        "label_spreading",
        family="graph_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Zhou et al. 2004 Learning with Local and Global Consistency",
    ),
    "label_propagation": _cap(
        "label_propagation",
        family="graph_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Zhu & Ghahramani 2002 Label Propagation",
    ),
    "self_training_lr": _cap(
        "self_training_lr",
        family="self_training",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Yarowsky 1995 / sklearn SelfTrainingClassifier",
    ),
    "self_training_xgboost": _cap(
        "self_training_xgboost",
        family="self_training",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Self-training with XGBoost base",
    ),
    "self_training_lightgbm": _cap(
        "self_training_lightgbm",
        family="self_training",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Self-training with LightGBM base",
    ),
    "self_training_catboost": _cap(
        "self_training_catboost",
        family="self_training",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="paper_core",
        reference_paper="Self-training with CatBoost base",
    ),
    "rpl_lr": _cap(
        "rpl_lr",
        family="rpl",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="faithful_reimplementation",
        reference_paper="Robust Pseudo-Labeling (RPL) family",
    ),
    "rpl_lite_xgboost": _cap(
        "rpl_lite_xgboost",
        family="rpl",
        input_view="processed",
        uses_unlabeled_data=True,
        fidelity="faithful_reimplementation",
        reference_paper="Robust Pseudo-Labeling (RPL) lite + XGBoost",
    ),
    # ------------------------------------------------------------------
    # Legacy neural SSL
    # ------------------------------------------------------------------
    "sslae": _cap(
        "sslae",
        family="neural_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="novel_experimental",
        reference_paper="Supervised autoencoder style SSL baseline",
        needs_external_validation=False,
    ),
    "vime": _cap(
        "vime",
        family="neural_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="paper_core",
        reference_paper="Yoon et al. 2020 VIME",
    ),
    "scarf": _cap(
        "scarf",
        family="neural_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="paper_core",
        reference_paper="Bahri et al. 2022 SCARF",
    ),
    "vime_lite": _cap(
        "vime_lite",
        family="neural_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="novel_experimental",
        reference_paper="VIME lite ablation (not full VIME)",
    ),
    # ------------------------------------------------------------------
    # TFM frozen / supervised baselines
    # ------------------------------------------------------------------
    "tabpfn3": _cap(
        "tabpfn3",
        family="tfm_frozen",
        input_view="raw",
        uses_unlabeled_data=False,
        device="gpu",
        fidelity="official",
        reference_paper="Hollmann et al. TabPFN / TabPFN-3",
        env="ssl-tfm",
    ),
    "tabiclv2": _cap(
        "tabiclv2",
        family="tfm_frozen",
        input_view="raw",
        uses_unlabeled_data=False,
        device="gpu",
        fidelity="official",
        reference_paper="TabICL / TabICLv2",
        env="ssl-tfm",
    ),
    "tabpfn3_self_training": _cap(
        "tabpfn3_self_training",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Iterative hard-label self-training with TabPFN-3",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_self_training": _cap(
        "tabiclv2_self_training",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Iterative hard-label self-training with TabICLv2",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    # ------------------------------------------------------------------
    # TFM SSL core
    # ------------------------------------------------------------------
    "tabpfn3_pl_one_shot": _cap(
        "tabpfn3_pl_one_shot",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="One-shot TFM pseudo-label context (TabPFN-3)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_pl_one_shot": _cap(
        "tabiclv2_pl_one_shot",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="One-shot TFM pseudo-label context (TabICLv2)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabpfn3_loop_risk": _cap(
        "tabpfn3_loop_risk",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Hard-label risk-controlled TFM looping (not soft LoopTabFM)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_loop_risk": _cap(
        "tabiclv2_loop_risk",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Hard-label risk-controlled TFM looping (not soft LoopTabFM)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabpfn3_cast": _cap(
        "tabpfn3_cast",
        family="tfm_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="CAST-style density adjustment adapted to TabPFN-3",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_cast": _cap(
        "tabiclv2_cast",
        family="tfm_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="CAST-style density adjustment adapted to TabICLv2",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tfm_consensus_context_tabiclv2": _cap(
        "tfm_consensus_context_tabiclv2",
        family="tfm_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TabPFN-3 ⊕ TabICLv2 consensus into TabICLv2 context",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabpfn3_teacher_catboost": _cap(
        "tabpfn3_teacher_catboost",
        family="tfm_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TabPFN-3 teacher → CatBoost student",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_teacher_catboost": _cap(
        "tabiclv2_teacher_catboost",
        family="tfm_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TabICLv2 teacher → CatBoost student",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tfm_consensus_catboost": _cap(
        "tfm_consensus_catboost",
        family="tfm_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TFM consensus pseudo-labels → CatBoost",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    # ------------------------------------------------------------------
    # Label-shift / prior adjustment
    # ------------------------------------------------------------------
    "tabpfn3_unlabeled_prior_adjustment": _cap(
        "tabpfn3_unlabeled_prior_adjustment",
        family="tfm_prior",
        input_view="raw",
        uses_unlabeled_data=True,
        protocol="inductive",
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Inductive unlabeled-pool prior adjustment for TabPFN-3",
        env="ssl-tfm",
    ),
    "tabpfn3_distpfn_transductive": _cap(
        "tabpfn3_distpfn_transductive",
        family="tfm_prior",
        input_view="raw",
        uses_unlabeled_data=True,
        protocol="transductive",
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="DistPFN placeholder — currently unsupported_faithful_distpfn_unavailable",
        env="ssl-tfm",
    ),
    # ------------------------------------------------------------------
    # Geometric / representation SSL (non-TFM)
    # ------------------------------------------------------------------
    "laplacian_linear": _cap(
        "laplacian_linear",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="paper_core",
        reference_paper="Laplacian-regularized linear classifier",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "laplacian_mlp": _cap(
        "laplacian_mlp",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="paper_core",
        reference_paper="Laplacian-regularized MLP",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "laplacian_ssl": _cap(
        "laplacian_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="paper_core",
        reference_paper="Sparse Laplacian-regularized MLP",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "prototype_alignment_ssl": _cap(
        "prototype_alignment_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Class-conditional prototype alignment SSL",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "embedding_alignment_ssl": _cap(
        "embedding_alignment_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Confidence-weighted class-conditional embedding alignment",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "retrieval_attention_ssl": _cap(
        "retrieval_attention_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Retrieval attention over unlabeled training memory",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "unlabeled_attention_ssl": _cap(
        "unlabeled_attention_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Retrieval attention over labeled and unlabeled training memory",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "geometric_attention_supervised": _cap(
        "geometric_attention_supervised",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=False,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Geometric attention ablation: supervised only",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "geometric_attention_laplacian": _cap(
        "geometric_attention_laplacian",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Geometric attention ablation: +Laplacian",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "geometric_attention_prototype": _cap(
        "geometric_attention_prototype",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Geometric attention ablation: +prototype",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "geometric_attention_retrieval": _cap(
        "geometric_attention_retrieval",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Geometric attention ablation: +retrieval",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "geometric_attention_ssl": _cap(
        "geometric_attention_ssl",
        family="geometric_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Combined geometric-attention SSL",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    # ------------------------------------------------------------------
    # TFM-conditioned geometric adapters
    # ------------------------------------------------------------------
    "tabpfn3_laplacian_adapter": _cap(
        "tabpfn3_laplacian_adapter",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Frozen TabPFN-3 + Laplacian adapter",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_laplacian_adapter": _cap(
        "tabiclv2_laplacian_adapter",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Frozen TabICLv2 + Laplacian adapter",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabpfn3_geometric_attention": _cap(
        "tabpfn3_geometric_attention",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Frozen TabPFN-3 + geometric attention adapter",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_geometric_attention": _cap(
        "tabiclv2_geometric_attention",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Frozen TabICLv2 + geometric attention adapter",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    # ------------------------------------------------------------------
    # Modern non-TFM SSL
    # ------------------------------------------------------------------
    "cast_catboost": _cap(
        "cast_catboost",
        family="modern_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="any",
        fidelity="faithful_reimplementation",
        reference_paper="CAST with CatBoost base",
        needs_external_validation=True,
        env="ssl-core",
    ),
    "cast_lightgbm": _cap(
        "cast_lightgbm",
        family="modern_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        device="cpu",
        fidelity="faithful_reimplementation",
        reference_paper="CAST with LightGBM base",
        needs_external_validation=True,
        env="ssl-core",
    ),
    "stunt": _cap(
        "stunt",
        family="modern_ssl",
        input_view="raw",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="official",
        reference_paper="STUNT",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "seba": _cap(
        "seba",
        family="modern_ssl",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="faithful_reimplementation",
        reference_paper="SeBA",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "d2r2_c": _cap(
        "d2r2_c",
        family="modern_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        protocol="inductive",
        device="gpu",
        fidelity="faithful_reimplementation",
        reference_paper="Inductive D2R2-c",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "d2r2c_inductive": _cap(
        "d2r2c_inductive",
        family="modern_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        protocol="inductive",
        device="gpu",
        fidelity="faithful_reimplementation",
        reference_paper="Inductive D2R2-c (alias of d2r2_c)",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "d2r2_transductive": _cap(
        "d2r2_transductive",
        family="modern_ssl",
        input_view="processed",
        uses_unlabeled_data=True,
        protocol="transductive",
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="Transductive D2R2 exploratory (excluded from inductive ranking)",
        needs_external_validation=True,
        env="ssl-representation",
    ),
    "tabiclv2_predfeat_laplacian_adapter": _cap(
        "tabiclv2_predfeat_laplacian_adapter",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TabICLv2 prediction-feature Laplacian adapter (not embedding)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
    "tabiclv2_predfeat_geometric": _cap(
        "tabiclv2_predfeat_geometric",
        family="tfm_geometric",
        input_view="both",
        uses_unlabeled_data=True,
        device="gpu",
        fidelity="novel_experimental",
        reference_paper="TabICLv2 prediction-feature geometric adapter (not embedding)",
        needs_external_validation=True,
        env="ssl-tfm",
    ),
}


METHOD_GROUPS: dict[str, list[str]] = {
    "tfm_frozen": [
        "tabpfn3",
        "tabiclv2",
    ],
    "tfm_ssl_core": [
        "tabpfn3_pl_one_shot",
        "tabiclv2_pl_one_shot",
        "tabpfn3_loop_risk",
        "tabiclv2_loop_risk",
        "tabpfn3_cast",
        "tabiclv2_cast",
        "tfm_consensus_context_tabiclv2",
        "tabpfn3_teacher_catboost",
        "tabiclv2_teacher_catboost",
        "tfm_consensus_catboost",
    ],
    "modern_ssl": [
        "cast_catboost",
        "cast_lightgbm",
        "stunt",
        "seba",
        "d2r2_c",
    ],
    "geometric_ssl_ablation": [
        "laplacian_linear",
        "laplacian_mlp",
        "prototype_alignment_ssl",
        "retrieval_attention_ssl",
        "geometric_attention_supervised",
        "geometric_attention_laplacian",
        "geometric_attention_prototype",
        "geometric_attention_retrieval",
        "geometric_attention_ssl",
    ],
    "tfm_geometric": [
        "tabpfn3_laplacian_adapter",
        "tabiclv2_laplacian_adapter",
        "tabpfn3_geometric_attention",
        "tabiclv2_geometric_attention",
    ],
    "transductive_exploratory": [
        "tabpfn3_distpfn_transductive",
        "d2r2_transductive",
    ],
}


def get_capabilities(method: str) -> MethodCapabilities:
    if method not in METHOD_CAPABILITIES:
        raise UnsupportedMethodError(
            method,
            f"Unknown method '{method}' (not in METHOD_CAPABILITIES).",
            status="unsupported_unknown_method",
        )
    return METHOD_CAPABILITIES[method]


def resolve_method_group(group: str) -> list[str]:
    if group not in METHOD_GROUPS:
        raise KeyError(
            f"Unknown method group '{group}'. "
            f"Available: {sorted(METHOD_GROUPS)}"
        )
    return list(METHOD_GROUPS[group])


def list_methods(*, protocol: ProtocolKind | None = None) -> list[str]:
    names = sorted(METHOD_CAPABILITIES)
    if protocol is None:
        return names
    return [n for n in names if METHOD_CAPABILITIES[n].protocol == protocol]


def assert_groups_consistent() -> None:
    """Raise if any group entry is missing from the capability registry."""
    missing: list[str] = []
    for group, methods in METHOD_GROUPS.items():
        for name in methods:
            if name not in METHOD_CAPABILITIES:
                missing.append(f"{group}:{name}")
    if missing:
        raise RuntimeError(f"METHOD_GROUPS references unknown methods: {missing}")


assert_groups_consistent()
