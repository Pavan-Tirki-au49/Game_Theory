# test_utils_svm.py — Diagnostic runner for SVM utilities

import os
import sys
import traceback

print("== test_utils_svm.py — Diagnostic run ==")
print("Python executable:", sys.executable)
print("Working dir:", os.getcwd())
print("Listing files in working dir:")
for fn in sorted(os.listdir(".")):
    print("  ", fn)

# show the local test image path (from workspace)
local_test_image = "/mnt/data/Screenshot 2025-11-20 234737.png"
print("Local test image path (for reference):", local_test_image)
print()

# Try to import utils and call the SVM helpers
try:
    import utils
    print("Imported utils from:", utils.__file__)
except Exception as e:
    print("FAILED to import utils.py:")
    traceback.print_exc()
    sys.exit(1)

# show whether model file exists
model_file = getattr(utils, "MODEL_FILENAME", "strategy_svm_model.joblib")
print("Model filename expected:", model_file)
print("Model exists on disk?:", os.path.exists(model_file))

# Try to call training helper (train_default_svm)
try:
    if hasattr(utils, "train_default_svm"):
        print("\nCalling utils.train_default_svm() ...")
        path = utils.train_default_svm(save_path=model_file)
        print("train_default_svm returned path:", path)
        print("Now model exists?:", os.path.exists(path))
    else:
        print("utils.train_default_svm not found.")
except Exception as e:
    print("Error while training model:")
    traceback.print_exc()

# Try to load model via utils.load_strategy_svm
try:
    if hasattr(utils, "load_strategy_svm"):
        print("\nCalling utils.load_strategy_svm() .")
        mdl = utils.load_strategy_svm(path=model_file)
        print("Loaded model object type:", type(mdl))

        # if sklearn pipeline, show classes_ and sample predict_proba
        if hasattr(mdl, "classes_"):
            print("Model classes_:", getattr(mdl, "classes_", None))

        if hasattr(mdl, "predict_proba"):
            sample_texts = ["i won't do business", "i will do business", "50-50 maybe"]
            print("Predict_proba on sample_texts:", sample_texts)
            probs = mdl.predict_proba(sample_texts)
            print("-> probs shape:", getattr(probs, "shape", None))
            print(probs)
    else:
        print("utils.load_strategy_svm not found.")
except Exception as e:
    print("Error while loading model:")
    traceback.print_exc()

# Try preview_removals_from_db (won't modify DB)
try:
    if hasattr(utils, "preview_removals_from_db") and hasattr(utils, "get_conn"):
        print("\nCalling preview_removals_from_db using utils.get_conn(...) ...")
        conn = utils.get_conn()
        try:
            flagged = utils.preview_removals_from_db(conn, threshold=0.8)
            print("preview_removals_from_db returned DataFrame with shape:", getattr(flagged, "shape", None))
            print(flagged.head(10).to_string(index=False))
        finally:
            conn.close()
    else:
        print("preview_removals_from_db or get_conn not available in utils.")
except Exception as e:
    print("Error while running preview_removals_from_db:")
    traceback.print_exc()

print("\n== Diagnostic run complete ==")
