#!/usr/bin/env python3
"""
augment_doc.py — Alias / wrapper untuk doc_augmented.py
Menjamin kompatibilitas apakah dipanggil via augment_doc.py maupun doc_augmented.py.
"""
from doc_augmented import generate_augmented_dataset, main

if __name__ == "__main__":
    main()
