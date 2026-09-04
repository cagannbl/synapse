import os
import pytest
from pathlib import Path
from synapse.lexer.lexer import Lexer
from synapse.lexer.token import TokenType
from synapse.parser.parser import Parser, parse_source
from synapse.parser.ast_nodes import (
    Program, VarDeclStmt, FunctionDef, BinaryExpr, PipeExpr,
    TensorLiteralExpr, IfStmt, ReturnStmt
)
from synapse.compiler.cache import ASTCache, clean_cache


def test_cache_set_and_get(tmp_path):
    """Test setting and retrieving an AST from ASTCache."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    source = "let learning_rate = 0.001\nconst BATCH_SIZE = 64"
    ast = parse_source(source, use_cache=False)

    cache.set(source, ast)
    retrieved = cache.get(source)

    assert retrieved is not None
    assert isinstance(retrieved, Program)
    assert len(retrieved.statements) == 2
    assert isinstance(retrieved.statements[0], VarDeclStmt)
    assert retrieved.statements[0].name == "learning_rate"
    assert retrieved.statements[1].name == "BATCH_SIZE"


def test_cache_hit_and_miss_counters(tmp_path):
    """Test cache hit and miss counting logic."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    source = "fn compute(x):\n    return x * 2"

    # 1. Miss on uncached source
    assert cache.get(source) is None
    assert cache.misses == 1
    assert cache.hits == 0

    # 2. Set and check hit (from memory)
    ast = parse_source(source, use_cache=False)
    cache.set(source, ast)
    hit1 = cache.get(source)
    assert hit1 is not None
    assert cache.hits == 1

    # 3. Hit from disk after clearing in-memory LRU
    cache._memory_cache.clear()
    hit2 = cache.get(source)
    assert hit2 is not None
    assert cache.hits == 2
    assert cache.misses == 1


def test_file_change_invalidation(tmp_path):
    """Test that modifying source code invalidates cache via SHA-256 hash change."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    source_v1 = "let model_version = 1"
    source_v2 = "let model_version = 2"

    ast_v1 = parse_source(source_v1, use_cache=False)
    cache.set(source_v1, ast_v1)

    # v1 is cached
    assert cache.get(source_v1) is not None

    # v2 is a different hash, should be a miss (invalidation of outdated code)
    assert cache.get(source_v2) is None

    # Now cache v2
    ast_v2 = parse_source(source_v2, use_cache=False)
    cache.set(source_v2, ast_v2)

    assert cache.get(source_v1).statements[0].value.value == 1
    assert cache.get(source_v2).statements[0].value.value == 2


def test_file_path_resolution_and_invalidation(tmp_path):
    """Test ASTCache resolving source from filename and detecting file edits."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    file_path = tmp_path / "script.syn"

    # Write version 1
    file_path.write_text("let x = 10", encoding="utf-8")
    ast1 = parse_source("", filename=str(file_path), use_cache=True, cache=cache)
    assert ast1.statements[0].value.value == 10
    assert cache.misses == 1

    # Second read should hit cache
    ast1_cached = parse_source("", filename=str(file_path), use_cache=True, cache=cache)
    assert ast1_cached.statements[0].value.value == 10
    assert cache.hits == 1

    # Edit file on disk (version 2)
    file_path.write_text("let x = 99", encoding="utf-8")
    ast2 = parse_source("", filename=str(file_path), use_cache=True, cache=cache)
    assert ast2.statements[0].value.value == 99
    assert cache.misses == 2  # New hash caused cache miss & fresh parse


def test_cache_clean(tmp_path):
    """Test cleaning disk cache files and clearing memory cache."""
    cache_dir = tmp_path / ".syn_cache"
    cache = ASTCache(cache_dir=cache_dir)

    for i in range(5):
        src = f"let var_{i} = {i}"
        ast = parse_source(src, use_cache=False)
        cache.set(src, ast)

    # Check files created on disk
    ast_files = list(cache_dir.glob("*.ast"))
    assert len(ast_files) == 5

    # Clean cache
    deleted = cache.clean()
    assert deleted == 5
    assert len(list(cache_dir.glob("*.ast"))) == 0
    assert len(cache._memory_cache) == 0

    # Cleaning already empty cache returns 0
    assert cache.clean() == 0

    # Static/Classmethod clean on non-existent directory returns 0
    assert ASTCache.clean(tmp_path / "non_existent") == 0


def test_ast_fidelity_and_integrity(tmp_path):
    """Test that cached and deserialized AST matches freshly parsed AST completely."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    complex_source = """
fn forward(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    let out = (x @ w + b) |> relu
    if out.shape[0] > 0:
        return out
    else:
        return x
"""
    fresh_ast = parse_source(complex_source, use_cache=False)
    cache.set(complex_source, fresh_ast)

    # Clear memory cache so it must load from disk via mmap + pickle protocol 5
    cache._memory_cache.clear()
    cached_ast = cache.get(complex_source)

    assert cached_ast is not None
    assert len(cached_ast.statements) == len(fresh_ast.statements)
    fn_fresh = fresh_ast.statements[0]
    fn_cached = cached_ast.statements[0]

    assert isinstance(fn_cached, FunctionDef)
    assert fn_cached.name == fn_fresh.name
    assert len(fn_cached.params) == len(fn_fresh.params)
    assert len(fn_cached.body) == len(fn_fresh.body)

    # Verify pipeline expression
    var_decl = fn_cached.body[0]
    assert isinstance(var_decl, VarDeclStmt)
    assert isinstance(var_decl.value, PipeExpr)
    assert isinstance(var_decl.value.left, BinaryExpr)
    assert var_decl.value.left.op == "+"

    # Verify if statement
    if_stmt = fn_cached.body[1]
    assert isinstance(if_stmt, IfStmt)
    assert len(if_stmt.then_branch) == 1
    assert len(if_stmt.else_branch) == 1


def test_mmap_protocol5_persistence_across_instances(tmp_path):
    """Test AST cache persistence across separate ASTCache instances using mmap."""
    cache_dir = tmp_path / ".syn_cache"
    source = "let embedding = tensor([1.0, 2.0, 3.0], requires_grad=false)"
    ast = parse_source(source, use_cache=False)

    # Instance 1 saves to disk
    cache1 = ASTCache(cache_dir=cache_dir)
    cache1.set(source, ast)

    # Instance 2 starts with empty memory cache
    cache2 = ASTCache(cache_dir=cache_dir)
    assert len(cache2._memory_cache) == 0

    # Instance 2 loads from disk via mmap protocol 5
    loaded = cache2.get(source)
    assert loaded is not None
    assert isinstance(loaded.statements[0].value, TensorLiteralExpr)
    # Check that memory cache is now populated in instance 2
    assert len(cache2._memory_cache) == 1


def test_lru_memory_eviction(tmp_path):
    """Test that in-memory cache enforces max_memory_entries and evicts LRU items."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache", max_memory_entries=3)

    sources = [f"let k_{i} = {i}" for i in range(5)]
    for src in sources:
        ast = parse_source(src, use_cache=False)
        cache.set(src, ast)

    # Memory cache should not exceed max_memory_entries=3
    assert len(cache._memory_cache) == 3

    # All 5 files exist on disk
    assert len(list((tmp_path / ".syn_cache").glob("*.ast"))) == 5

    # Oldest entry (sources[0]) was evicted from memory, but get() still retrieves it from disk
    old_hash = ASTCache.compute_hash(sources[0])
    assert old_hash not in cache._memory_cache

    retrieved = cache.get(sources[0])
    assert retrieved is not None
    # Now sources[0] should be back in memory LRU
    assert old_hash in cache._memory_cache


def test_lexer_fast_path_equivalence():
    """Test that Lexer with fast-path produces exact same tokens as without fast-path."""
    sample_code = """
# Comprehensive Synapse test code
fn train_step(X: Tensor, y: Tensor, epochs: int = 100, lr: float = 0.001) -> float:
    let w = tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=true)
    let b = tensor([0.0, 0.0], requires_grad=true)
    let total_loss = 0.0
    const SCALE = 1.5e-3
    const BIG_NUM = 2E+4

    for epoch in range(epochs):
        let pred = (X @ w + b) |> relu
        let loss = mse_loss(pred, y)
        total_loss += loss.item()

        if total_loss > 1000.0:
            let flag = true
            break
        elif total_loss == 0.0:
            let status = none
        else:
            let flag = false

    return total_loss
"""
    lexer_fast = Lexer(sample_code, use_fast_path=True)
    tokens_fast = lexer_fast.tokenize()

    lexer_slow = Lexer(sample_code, use_fast_path=False)
    tokens_slow = lexer_slow.tokenize()

    assert len(tokens_fast) == len(tokens_slow)
    for i, (tf, ts) in enumerate(zip(tokens_fast, tokens_slow)):
        assert tf.type == ts.type, f"Token type mismatch at index {i}: {tf.type} vs {ts.type}"
        assert tf.value == ts.value, f"Token value mismatch at index {i}: {tf.value} vs {ts.value}"
        assert tf.line == ts.line, f"Token line mismatch at index {i}: {tf.line} vs {ts.line}"
        assert tf.column == ts.column, f"Token col mismatch at index {i}: {tf.column} vs {ts.column}"

    # Verify scan_fast method on an individual token
    lexer_direct = Lexer("42.5e-2")
    tok = lexer_direct.scan_fast()
    assert tok is not None
    assert tok.type == TokenType.FLOAT
    assert tok.value == pytest.approx(0.425)


def test_parse_source_use_cache_toggle(tmp_path):
    """Test parse_source use_cache flag behavior."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache")
    source = "let alpha = 0.05"

    # 1. With use_cache=True (first call misses and sets, second call hits)
    ast1 = parse_source(source, use_cache=True, cache=cache)
    assert cache.misses == 1
    assert cache.hits == 0

    ast2 = parse_source(source, use_cache=True, cache=cache)
    assert cache.hits == 1
    assert ast1.statements[0].name == ast2.statements[0].name

    # 2. With use_cache=False (bypasses cache)
    ast3 = parse_source(source, use_cache=False, cache=cache)
    assert cache.hits == 1  # Hits count unchanged


def test_cache_corrupt_file_handling(tmp_path):
    """Test that corrupted cache files on disk do not crash the application."""
    cache_dir = tmp_path / ".syn_cache"
    cache = ASTCache(cache_dir=cache_dir)
    source = "let x = 42"

    ast = parse_source(source, use_cache=False)
    cache.set(source, ast)

    # Corrupt the disk file with random bytes
    key = ASTCache.compute_hash(source)
    cache_file = cache_dir / f"{key}.ast"
    cache_file.write_bytes(b"\x00\xffCORRUPTED_DATA_HEADER\x00")

    # Clear memory cache so it attempts to read disk
    cache._memory_cache.clear()

    # Should return None gracefully instead of raising unhandled exception
    res = cache.get(source)
    assert res is None
    assert cache.misses == 1


def test_disabled_cache(tmp_path):
    """Test that disabling cache skips all caching actions."""
    cache = ASTCache(cache_dir=tmp_path / ".syn_cache", enabled=False)
    source = "let a = 1"
    ast = parse_source(source, use_cache=False)

    cache.set(source, ast)
    assert not (tmp_path / ".syn_cache").exists()
    assert cache.get(source) is None
