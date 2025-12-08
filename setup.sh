mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \${PORT:-8501}\n\
" > ~/.streamlit/config.toml
