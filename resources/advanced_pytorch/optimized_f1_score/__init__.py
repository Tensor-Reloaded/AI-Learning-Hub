from optimized_f1_score.build import build



try:
    from optimized_f1_score.f1_macro_cpp import f1_macro
except ImportError as e:
    try:
        build()
        from optimized_f1_score.f1_macro_cpp import f1_macro
    except ImportError as e:
        print("Building Optimized F1 Score failed. Using Fallback")
        from optimized_f1_score.f1_macro_py import f1_macro


