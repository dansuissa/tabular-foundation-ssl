"""Capability registry for the 25 canonical benchmark methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.exceptions import UnsupportedMethodError

InputView = Literal["processed", "raw", "both"]
ProtocolKind = Literal["inductive", "transductive"]
DeviceKind = Literal["cpu", "gpu", "any"]
FidelityKind = Literal["official", "faithful_reimplementation", "paper_core", "novel_experimental"]
EnvKind = Literal["ssl-core", "ssl-tfm"]


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
    fidelity: FidelityKind,
    reference_paper: str | None = None,
    protocol: ProtocolKind = "inductive",
    device: DeviceKind = "cpu",
    supports_binary: bool = True,
    supports_multiclass: bool = True,
    upstream_commit: str | None = None,
    needs_external_validation: bool = False,
    env: EnvKind = "ssl-core",
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
    )


METHOD_CAPABILITIES: dict[str, MethodCapabilities] = {
    "logistic_regression": _cap("logistic_regression", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="paper_core", reference_paper="scikit-learn LogisticRegression"),
    "random_forest": _cap("random_forest", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="paper_core", reference_paper="Breiman 2001 Random Forests"),
    "xgboost": _cap("xgboost", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="official", reference_paper="Chen & Guestrin 2016 XGBoost"),
    "lightgbm": _cap("lightgbm", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="official", reference_paper="Ke et al. 2017 LightGBM"),
    "catboost": _cap("catboost", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="official", reference_paper="Prokhorenkova et al. 2018 CatBoost"),
    "mlp": _cap("mlp", family="supervised", input_view="processed", uses_unlabeled_data=False, fidelity="paper_core", reference_paper="scikit-learn MLPClassifier"),

    "label_spreading": _cap("label_spreading", family="graph_ssl", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Zhou et al. 2004"),
    "label_propagation": _cap("label_propagation", family="graph_ssl", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Zhu & Ghahramani 2002"),

    "self_training_lr": _cap("self_training_lr", family="self_training", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Classical self-training with logistic regression"),
    "self_training_xgboost": _cap("self_training_xgboost", family="self_training", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Classical self-training with XGBoost"),
    "self_training_lightgbm": _cap("self_training_lightgbm", family="self_training", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Classical self-training with LightGBM"),
    "self_training_catboost": _cap("self_training_catboost", family="self_training", input_view="processed", uses_unlabeled_data=True, fidelity="paper_core", reference_paper="Classical self-training with CatBoost"),

    "rpl_lr": _cap("rpl_lr", family="rpl", input_view="processed", uses_unlabeled_data=True, fidelity="faithful_reimplementation", reference_paper="Robust pseudo-labeling family"),
    "rpl_lite_xgboost": _cap("rpl_lite_xgboost", family="rpl", input_view="processed", uses_unlabeled_data=True, fidelity="faithful_reimplementation", reference_paper="Robust pseudo-labeling family with XGBoost"),

    "sslae": _cap("sslae", family="neural_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="novel_experimental", reference_paper="Supervised-autoencoder-style SSL baseline"),
    "vime": _cap("vime", family="neural_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="paper_core", reference_paper="Yoon et al. 2020 VIME"),
    "scarf": _cap("scarf", family="neural_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="paper_core", reference_paper="Bahri et al. 2022 SCARF"),

    "tabpfn3": _cap("tabpfn3", family="tfm_frozen", input_view="raw", uses_unlabeled_data=False, device="gpu", fidelity="official", reference_paper="TabPFN-3", env="ssl-tfm"),
    "tabiclv2": _cap("tabiclv2", family="tfm_frozen", input_view="raw", uses_unlabeled_data=False, device="gpu", fidelity="official", reference_paper="TabICL v2", env="ssl-tfm"),
    "tabpfn3_self_training": _cap("tabpfn3_self_training", family="tfm_ssl", input_view="raw", uses_unlabeled_data=True, device="gpu", fidelity="novel_experimental", reference_paper="Validation-guarded self-training with TabPFN-3", needs_external_validation=True, env="ssl-tfm"),
    "tabiclv2_self_training": _cap("tabiclv2_self_training", family="tfm_ssl", input_view="raw", uses_unlabeled_data=True, device="gpu", fidelity="novel_experimental", reference_paper="Validation-guarded self-training with TabICL v2", needs_external_validation=True, env="ssl-tfm"),

    "laplacian_ssl": _cap("laplacian_ssl", family="geometric_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="novel_experimental", reference_paper="Sparse Laplacian-regularized neural classifier", needs_external_validation=True, env="ssl-tfm"),
    "unlabeled_attention_ssl": _cap("unlabeled_attention_ssl", family="geometric_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="novel_experimental", reference_paper="Retrieval attention over labeled and unlabeled training memory", needs_external_validation=True, env="ssl-tfm"),
    "embedding_alignment_ssl": _cap("embedding_alignment_ssl", family="geometric_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="novel_experimental", reference_paper="Class-conditional embedding alignment", needs_external_validation=True, env="ssl-tfm"),
    "geometric_attention_ssl": _cap("geometric_attention_ssl", family="geometric_ssl", input_view="processed", uses_unlabeled_data=True, device="any", fidelity="novel_experimental", reference_paper="Combined geometric and retrieval SSL", needs_external_validation=True, env="ssl-tfm"),
}

HISTORICAL_METHODS = [
    "logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "mlp",
    "label_spreading", "label_propagation",
    "self_training_lr", "self_training_xgboost", "self_training_lightgbm", "self_training_catboost",
    "rpl_lr", "rpl_lite_xgboost", "sslae", "vime", "scarf",
]
TFM_FROZEN_METHODS = ["tabpfn3", "tabiclv2"]
FOCUSED_SSL_METHODS = [
    "tabpfn3_self_training", "tabiclv2_self_training", "laplacian_ssl",
    "unlabeled_attention_ssl", "embedding_alignment_ssl", "geometric_attention_ssl",
]
METHOD_GROUPS: dict[str, list[str]] = {
    "historical_methods": HISTORICAL_METHODS,
    "tfm_frozen": TFM_FROZEN_METHODS,
    "focused_tfm_ssl": FOCUSED_SSL_METHODS,
    "canonical_methods": HISTORICAL_METHODS + TFM_FROZEN_METHODS + FOCUSED_SSL_METHODS,
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
        raise KeyError(f"Unknown method group '{group}'. Available: {sorted(METHOD_GROUPS)}")
    return list(METHOD_GROUPS[group])


def list_methods(*, protocol: ProtocolKind | None = None) -> list[str]:
    names = sorted(METHOD_CAPABILITIES)
    if protocol is None:
        return names
    return [name for name in names if METHOD_CAPABILITIES[name].protocol == protocol]


def assert_groups_consistent() -> None:
    missing = [
        f"{group}:{name}"
        for group, methods in METHOD_GROUPS.items()
        for name in methods
        if name not in METHOD_CAPABILITIES
    ]
    if missing:
        raise RuntimeError(f"METHOD_GROUPS references unknown methods: {missing}")


assert_groups_consistent()
