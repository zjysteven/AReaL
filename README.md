# AReaL + AIgiSE

## Setup

```bash
git submodule update --init --recursive

# we use uv workspace to simultaneously install dependencies for AReaL and AIgiSE
uv sync --extra cuda
```
Then, follow AIgiSE's README to prepare CodeQL.

## Example runs

`run_secodeplt.sh`: run the secodeplt evaluation with gemini-3-pro-preview model.
`run.sh`: run areal training on secodeplt with Qwen3-8B model.