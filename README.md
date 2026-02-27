

# Ejecutar todos los ejemplos 
for f in examples/*.py; do echo "=== $f ===" && uv run python "$f"; done
