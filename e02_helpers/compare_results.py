def compare_results(before_results, after_results):
    if len(before_results) != len(after_results):
        raise ValueError("Before and after results must use the same examples.")
    for index, (before, after) in enumerate(zip(before_results, after_results), start=1):
        if before["reference"] != after["reference"]:
            raise ValueError("Before and after examples do not match.")
        print(f"\nExample {index}")
        print("Clinician:", before["reference"])
        print("Before:", before["reply"])
        print("After:", after["reply"])
        if not before["ended"] or not after["ended"]:
            print("A reply reached its token limit without an end marker.")
