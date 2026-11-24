"""Explicit native recipe dispatch; native_rl never implies a particular algorithm."""

from importlib import import_module

RECIPES = {
    "rpo_ufo_fb_native": "ufo",
    "rpo_pbfm_fb_depth_native": "pbfm",
}


def is_native_template(template_id: str | None) -> bool:
    return template_id in RECIPES


def service_for(template_id: str):
    try:
        recipe = RECIPES[template_id]
    except KeyError as exc:
        raise ValueError(f"Unregistered native training recipe: {template_id}") from exc
    return import_module(f"eb.{recipe}_training_service")


def recipe_for(template_id: str):
    try:
        recipe = RECIPES[template_id]
    except KeyError as exc:
        raise ValueError(f"Unregistered native training recipe: {template_id}") from exc
    return import_module(f"eb.{recipe}_training")
