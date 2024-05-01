import copy
from montara.des_tile import _set_catalog_sampler_rng_num


def test_set_catalog_sampler_rng_num():
    cfg = {"blah": {"foo": [{"type": "catalog_sampler_value"}]}, "blahblah": {"type": "blah"}, "bar": {"type": "catalog_sampler_value"}}
    new_cfg = _set_catalog_sampler_rng_num(copy.deepcopy(cfg), 1)
    assert new_cfg == {
        "blah": {
            "foo": [{"type": "catalog_sampler_value", "rng_num": 1}]
            },
        "blahblah": {
            "type": "blah"
        },
        "bar": {
            "type": "catalog_sampler_value",
            "rng_num": 1
        }
    }

    cfg = {"blah": {"foo": [{"type": "catalog_sampler_value"}]}, "blahblah": {"type": "blah"}, "bar": {"type": "catalog_sampler_value"}}
    cfg = _set_catalog_sampler_rng_num(cfg, 1)
    assert cfg == {
        "blah": {
            "foo": [{"type": "catalog_sampler_value", "rng_num": 1}]
            },
        "blahblah": {
            "type": "blah"
        },
        "bar": {
            "type": "catalog_sampler_value",
            "rng_num": 1
        }
    }

