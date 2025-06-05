# The Society of HiveMind

🔥 Accepted at [ICSI 2025](https://www.iasei.org/icsi2025/) for oral presentation.

🐝 **HiveMind is a graph-based framework for AI foundation model agents, providing two high-level features:**

* It lets you build artificial agents from graphs.
* It enables the customized and automatic self-organization of agent swarms with self-improvement capabilities.

<p align="left">
<a href=""><img src="assets/HiveMind.png" alt="HiveMind" width="600px"></a>
</p>

## Edge optimization example

Here is the edge optimization process that updates edge probabilities toward improvement of the benchmark score. Notice that within an agent, the edges are fixed, whereas the inter-agent connections are getting optimized towards either edge pruning (value 0, red) or creation (value 1, yellow).

<img src="assets/edge_opt_base.gif" alt="Edge optimization" width="400">


## About HiveMind

At a granular level, HiveMind is a library based on the excellent project [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) that includes the following components: 


| Module | Description |
| ---- | --- |
| [**swarm.environment**](swarm/environment) | Domain-specific operations, agents, tools, and tasks |
| [**swarm.graph**](swarm/graph) | Graph-related functions for creating and executing agent graphs and swarm composite graphs |
| [**swarm.llm**](swarm/llm) | Interface for selecting LLM backends and calculating their operational costs |
| [**swarm.memory**](swarm/memory) | Index-based memory |
| [**swarm.optimizer**](swarm/optimizer) | Optimization algorithms designed to enhance agent performance and overall swarm efficiency |


## Quickstart

**Install packages**
```
conda create -n swarm python=3.10
conda activate swarm
pip install poetry
poetry install
```

<!-- **You should add API keys in `.env.template` and change its name to `.env`**

```python
OPENAI_API_KEY="" # for OpenAI LLM backend
SEARCHAPI_API_KEY="" # for Web Search
``` -->

<!-- **Getting started with HiveMind is easy. Quickly run a predefined swarm**

```python
from swarm.graph.swarm import Swarm

swarm = Swarm(["IO", "IO", "IO"], "gaia")
task = "What is the capital of Jordan?"
inputs = {"task": task}
answer = await swarm.arun(inputs)
```

**or make use of tools, such as the file analyzer**

```python
from swarm.graph.swarm import Swarm
swarm = Swarm(["IO", "TOT"], "gaia")
task = "Tell me more about this image and summarize it in 3 sentences."
files = ["./datasets/demos/js.png"]
inputs = {"task": task, "files": files}
danswer = swarm.run(inputs)
``` -->

## Experiments

For a more advanced usage of this framework, see our [experiments](https://github.com/nmamie/HiveLLM/tree/experimental_ga/experiments). 🚀🚀🚀


<!-- ## Class diagram

<img src="assets/class_diagram.png" alt="Edge optimization" width="700">

## Example of a Swarm

<img src="assets/swarm_v3.png" alt="Edge optimization" width="500"> -->

## Acknowledgements

We acknowledge the excellent projects by [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) and [Evolutionary-Reinforcement-Learning](https://github.com/ShawK91/Evolutionary-Reinforcement-Learning) that largely inspired the development of the Society of HiveMind.


<!-- ## Contributors -->

<!-- 
## Citation
Please cite our paper if you find the library useful or interesting.
```
@article{}
``` -->
