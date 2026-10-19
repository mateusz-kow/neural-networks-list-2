## Report

The report is available in the REPORT.md file

## How to run

### Sync the dependencies

```bash
uv sync --extra dev
```

### Configure and run pre-commit

```bash
uv run prek install
uv run prek run --all-files
```

### Run the project

```bash
uv run main
```

### Generate the PDF report

```bash
pandoc REPORT.md \
  -o SN_L2_Kowalski_Mateusz.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  -V fontsize=11pt \
  -V lang=pl-PL
```
