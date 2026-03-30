#!/usr/bin/env bash
# 使用项目内 .neo4j_homebrew（Homebrew bottle 解包）与 .jdk21（便携 Temurin 21）启动 Neo4j。
# 配置目录：datasets/neo4j_runtime/conf
# 数据目录：datasets/neo4j_data/
#
# 首次部署：
#   1. 提取 Neo4j bottle：见 SteelDig 文档或手动 tar -xzf ~/Library/Caches/Homebrew/downloads/*neo4j*.all.bottle.tar.gz
#   2. 下载 JDK21 tar 到项目 .jdk21（与 json_entities_to_neo4j 说明一致）
#   3. neo4j-admin dbms set-initial-password ...（仅在全新 data 目录时）
#
# 用法：./scripts/start_neo4j_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export JAVA_HOME="${JAVA_HOME:-$ROOT/.jdk21/Contents/Home}"
export NEO4J_HOME="${NEO4J_HOME:-$ROOT/.neo4j_homebrew/neo4j/2026.02.3/libexec}"
export NEO4J_CONF="${NEO4J_CONF:-$ROOT/datasets/neo4j_runtime/conf}"
mkdir -p "$ROOT/datasets/neo4j_data/"{data,logs,import,run,transactions}
exec "$NEO4J_HOME/bin/neo4j" console
