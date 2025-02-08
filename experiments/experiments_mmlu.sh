#####################BASE EXPERIMENTS#####################

PYTHONPATH=. python experiments/run_mmlu.py --mode=DirectAnswer

PYTHONPATH=. python experiments/run_mmlu.py --mode=COT

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --adversarial --mode=FullConnectedSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --random-string --mode=FullConnectedSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=FullConnectedSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --adversarial --mode=RandomSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --random-string --mode=RandomSwarm

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=RandomSwarm

#####################OPTIMIZATION EXPERIMENTS#####################

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --adversarial --beta 0.0 --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --random-string --beta 0.0 --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --beta 0.0 --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --adversarial --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --random-string --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --optimizer=ga --adversarial --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga --random-string --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga --num-iterations 60

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=3 --mode=OptimizedSwarm --optimizer=ga --adversarial --num-iterations 60 --lr 0.005

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga --random-string --num-iterations 60 --lr 0.005

PYTHONPATH=. python experiments/run_mmlu.py --num-truthful-agents=6 --mode=OptimizedSwarm --optimizer=ga --num-iterations 60 --lr 0.005