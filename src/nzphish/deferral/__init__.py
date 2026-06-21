"""Class-conditional selective prediction / deferral layer.

Learns per-class confidence thresholds so the classifier abstains on borderline
samples and routes them for human review. Extends LW-CCSD (rPPG AF screening) and
OACSP (retinal DR grading) prior work by the same author to binary
phishing/legitimate classification under NZ threat-model coverage constraints.
"""
