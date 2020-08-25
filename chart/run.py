import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import subprocess

def create_dataframe(results):
    """Parses the \n delimited JSON lines into a pandas DataFrame"""
    rows = []
    print(results)
    for line in results.splitlines():
        line = json.loads(line)
        if line.get("avg") == None: continue
        
        rows.append({
            "timestamp": line["timestamp"],
            "impl": line["spans"][2]["kind"],
            "task": line["spans"][0]["task"],
            "trial": line["spans"][1]["trial_num"],
            "threads": line["span"]["threads"],
            "mix": line["span"]["mix"],
            "avg": int(line["avg"].strip('ns')),
            "ops": line["ops"],
            "took": line["took"],
            "_debug": line["message"]  
        })
            
    return pd.DataFrame.from_records(rows)


def save_plots(df):
    """
    Saves a comparison plot in the current directory for each Mix present being compared. The
    filename template is `avg_performance_{task}.png`.
    """

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)

    for task, task_df in df.groupby('task'):
        
        fig, ax = plt.subplots(figsize=(16,12))
        
        title = task.replace("_", " ")
        title = title.title()
        # substitute with actual params used when those
        # become an option
        subtitle = f"random seeds={5}; params=default" 

        # Plot each implementation in the chart 
        for label, group in task_df.groupby('impl'):
            
            see = group[['threads', 'avg']]
            see = see.groupby('threads').mean().reset_index()
            see.plot(
                x="avg", 
                y="threads", 
                ax=ax, 
                label=label, #.strip("main::adapters::").strip("<u64>"),
                style='x', # '.--'
                ms=10
            )
        
        yticks = list(filter(lambda x: x % 2 == 0, list(df["threads"].unique())))
      
        
        fig.suptitle(title, y=0.97, fontsize=26)
            
        plt.title(subtitle, y=1.01, fontsize=22) #, fontsize=10)
        plt.legend()
        
        ax.set_yticks(yticks)
        ax.set_ylabel("threads", fontsize=24, labelpad=20)
        ax.set_xlabel("avg ns / op", fontsize=24, labelpad=20)

        fig.savefig(f"avg_performance_{task}.png")

def main():
    """
    Runs comparison benchmarks and generates plots for several preset
    workloads from bustle::Mix
    """
    print("Running `cargo run`")
    # Run the main cargo bin
    proc = subprocess.Popen(['cargo', 'run', '--release', '--bin', 'main'], 
           stdout=subprocess.PIPE, 
           stderr=subprocess.DEVNULL)
    
    # Capture stdout
    res, _ = proc.communicate()

    # Make a DataFrame
    df = create_dataframe(res)

    save_plots(df)

main()
