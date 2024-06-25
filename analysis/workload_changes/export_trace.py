from smart_pred.dataset.azure_trace_2019 import AzureFunction2019


ds = AzureFunction2019()
ds.load_and_cache_dataset()






def export_a_bursty_changes_trace():
    hash_function = "40d07b9935b0d765eec41cedf70847d06699c0df3e24e1216e0ca893e6e8f71d"

    trace = ds.get_all_invocation_trace_by_hash_function(
        hash_function=hash_function,
    )
    print(f"trace: {trace}")
    print(f"len(trace): {len(trace)}")
    # 保存到csv

    import pandas as pd
    df = pd.DataFrame(trace)
    df.to_csv(f"./{hash_function}.csv", index=False)



if __name__ == "__main__":
    export_a_bursty_changes_trace()