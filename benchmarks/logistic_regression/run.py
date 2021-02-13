import pandas as pd
from rich.progress import Progress
from river.compat import SKL2RiverClassifier
from river.compose import Pipeline
from river.evaluate import load_binary_clf_tracks
from river.linear_model import LogisticRegression
from river.optim import SGD
from river.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


LEARNING_RATE = 0.005

MODELS = {
    'River': Pipeline(
        StandardScaler(),
        LogisticRegression(optimizer=SGD(LEARNING_RATE))
    ),
    'scikit-learn': Pipeline(
        StandardScaler(),
        SKL2RiverClassifier(
            SGDClassifier(
                loss='log',
                learning_rate='constant',
                eta0=LEARNING_RATE,
                penalty='none'
            ),
            classes=[False, True]
        )
    )
}

def run():

    results = []
    tracks = load_binary_clf_tracks()
    n_checkpoints = 10

    with Progress() as progress:
        bar = progress.add_task("Models", total=len(MODELS))

        for model_name, model in MODELS.items():
            model_bar = progress.add_task(f"[green]{model_name}", total=len(tracks))

            for track in tracks:
                track_bar = progress.add_task(f"[cyan]{track.name}", total=n_checkpoints)

                for step in track.run(model, n_checkpoints=n_checkpoints):
                    step['Model'] = model_name
                    step['Track'] = track.name
                    results.append(step)
                    progress.advance(track_bar)
                progress.advance(model_bar)
            progress.advance(bar)

    return pd.DataFrame(results).set_index(['Model', 'Track', 'Step']).reset_index()



if __name__ == '__main__':
    results = run()

    with open('README.md', 'w') as f:
        print('# Logistic regression\n', file=f)

        print('## Final results\n', file=f)
        final = (
            results
            .sort_values('Step')
            .groupby(['Model', 'Track'])
            .last()
            .reset_index()
            .drop(columns=['Step'])
        )
        print(final.to_markdown(index=False), file=f)
        print('', file=f)

        print('## Full traces\n', file=f)
        print(results.to_markdown(index=False), file=f)
