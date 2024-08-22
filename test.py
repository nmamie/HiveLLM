import asyncio
from swarm.graph.swarm import Swarm

async def main():
	swarm = Swarm(["IO", "COT"], "gaia", model_name="custom")
	# task = "What is the capital of Jordan?"
	task = "Who is the current president of Switzerland?"
	inputs = {"task": task}
	answer = await swarm.arun(inputs)

asyncio.run(main())