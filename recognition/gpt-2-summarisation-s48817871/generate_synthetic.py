import openai
from datasets import load_dataset
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("--url", default="http://localhost:9931/v1", help="URL of OpenAI API compatible endpoint (Default is a local llama.cpp endpoint)")
parser.add_argument("--model", default= "Qwen 3.6 35B A3B IQ4_XS Speed", help="Model from API that should be used (Default is a specialied Qwen model a local llama.cpp endpoint)")
parser.add_argument("--key", default="key", help="API key for OpenAI API compatible endpoint (Default is \"key\")")
parser.add_argument("--limit-rows", default=10, type=int, help="Number of rows from BioLaySumm dataset to be rewritten. -1 for all rows.")
parser.add_argument("--output-csv", default="synthetic.csv", help="Output CSV of rewritten data (Default is \"synthetic.csv\")")

EFFORT = "none"
sem = asyncio.Semaphore(16)

args = parser.parse_args()

RADIOLOGY_REPORT = 'radiology_report'
LAYMAN_REPORT = 'layman_report'
REPORT_REWRITE_FORMAT = "Please rewrite the following in roughly the same length. Write nothing else. \"{}\""
REPORT_SUMMARISE_FORMAT = "Please rewrite and summarise the following, keep it roughly the same length, reword it to a 6th grade reading level, keep the summary third person. Write nothing else. \"{}\""

async def rewrite_and_summarise(report: str, client: openai.OpenAI, i: int, df: pd.DataFrame) -> None:
    async with sem:
        response = await client.responses.create(
            model=args.model,
            reasoning={"effort":EFFORT},
            input=REPORT_REWRITE_FORMAT.format(report),
        )
        rewrite = response.output_text
        response = await client.responses.create(
            model=args.model,
            reasoning={"effort":EFFORT},
            input=REPORT_SUMMARISE_FORMAT.format(report),
        )
        summary = response.output_text
        df.loc[i] = {RADIOLOGY_REPORT: rewrite, LAYMAN_REPORT: summary}

async def main():
    client = openai.AsyncOpenAI(
            base_url=args.url,
            api_key=args.key,
            )
    # force load of model if on llama cpp server
    if args.url == "http://localhost:9931/v1":
        client.responses.create(
            model=args.model,
            reasoning={"effort":EFFORT},
            input="hello",
        )
    bls = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="train")
    df = pd.DataFrame(columns=[RADIOLOGY_REPORT, LAYMAN_REPORT])
    if args.limit_rows != -1:
        limrows = min(bls.num_rows, args.limit_rows)
    else:
        limrows = bls.num_rows
    await tqdm_asyncio.gather(
        *(rewrite_and_summarise(str(bls[RADIOLOGY_REPORT][i]), client, i, df) for i in range(limrows))
            )
    df.to_csv(args.output_csv)

if __name__ == "__main__":
    asyncio.run(main())
