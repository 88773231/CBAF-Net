"""Bound the frozen replay's fold cache without changing its scientific checks."""

import ast
import copy
import hashlib
import inspect

from analysis import run_botsscl_policy_control as frozen


class SingleFoldCache(dict):
    def __setitem__(self, key, value):
        self.clear()
        super().__setitem__(key, value)


def replay(data_dir, temporal, partitions, dataset, root, output):
    # Keep the training-version file unchanged because completed protocols hash it.
    original = ast.parse(inspect.getsource(frozen.replay))
    adapted = copy.deepcopy(original)
    replacements = 0
    for node in ast.walk(adapted):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and [value.id if isinstance(value, ast.Name) else None
                     for value in node.targets[0].elts] == ["config", "cache", "audited"]
                and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 3
                and isinstance(node.value.elts[1], ast.Dict)
                and not node.value.elts[1].keys):
            node.value.elts[1] = ast.Call(
                func=ast.Name(id="SingleFoldCache", ctx=ast.Load()), args=[], keywords=[],
            )
            replacements += 1
    if replacements != 1:
        raise ValueError("the frozen replay cache initialization differs")
    ast.fix_missing_locations(adapted)
    namespace = {**frozen.replay.__globals__, "SingleFoldCache": SingleFoldCache}
    exec(compile(adapted, "<bounded-frozen-botsscl-replay>", "exec"), namespace)
    namespace["replay"](data_dir, temporal, partitions, dataset, root, output)
    return {
        "resource_change": "single_fold_cache_only",
        "cache_initializers_replaced": replacements,
        "original_replay_source_sha256": hashlib.sha256(
            inspect.getsource(frozen.replay).encode()
        ).hexdigest(),
        "adapted_ast_sha256": hashlib.sha256(ast.dump(adapted).encode()).hexdigest(),
        "training_rerun": False,
    }
