import asyncio


async def async_chain_run(chain, input_variables):
    result = await chain.arun(**input_variables)
    return result


async def run_chains(chains, input_variables_lst):
    tasks = [async_chain_run(chain, input_variables_lst[c]) for c, chain in enumerate(chains)]
    return await asyncio.gather(*tasks)


def get_chains_results(chains, input_variables_lst):
    return asyncio.run(run_chains(chains, input_variables_lst))


def get_chain_result(chain, input_variables):
    return chain.run(**input_variables)
