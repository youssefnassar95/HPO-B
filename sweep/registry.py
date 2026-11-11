"""The HPO methods a sweep can run, by name."""

from methods.generative_hpo import GenerativeHPO
from methods.random_search import RandomSearch


def _build_random(dim):
    return RandomSearch()


def _build_gen_vae(dim, n_batches=100, batch_size=16, dim_model=32):
    # One token is a configuration plus its response.
    return GenerativeHPO(
        num_tokens=dim + 1,
        dim_model=dim_model,
        n_batches=n_batches,
        batch_size=batch_size,
    )


BUILDERS = {
    "random": _build_random,
    "gen-vae": _build_gen_vae,
}

_instances = {}


def get_method(name, dim, options):
    """Return the method registered under `name`, building it on first use."""
    key = (name, dim)
    if key not in _instances:
        _instances[key] = BUILDERS[name](dim, **options)
    return _instances[key]
