"""WS4: training-signal study, post-training first. Are KU-derived representations a good training signal?
papers -> probe -> data -> qa-data -> decontam -> train-sft / train-grpo / train-tools -> eval -> grade -> results
-> report."""
from . import papers, data, qa_data, train, evaluate, results  # noqa: F401
from ..report import register_report

register_report("ws4", "WS4", "Training signal: post-training (SFT, GRPO, tool RL) on KUs vs full papers and controls")
