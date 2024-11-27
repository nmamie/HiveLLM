#####################BASE EXPERIMENTS#####################

PYTHONPATH=. python experiments/run_mmlu.py --mode=DirectAnswer

PYTHONPATH=. python experiments/run_mmlu.py --mode=COT

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=FullConnectedSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=RandomSwarm

#####################OPTIMIZATION EXPERIMENTS#####################

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --adversarial --beta 0.0

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --adversarial

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --random-string

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --optimizer=ga --adversarial

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga --random-string

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga