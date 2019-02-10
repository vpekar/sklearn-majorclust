A [scikit-learn](https://scikit-learn.org/stable/index.html) API for the [MajorClust](https://www.semanticscholar.org/paper/Document-Categorization-with-MAJORCLUST-Stein-Eissen/2380d838f03d0564631475904dc61e4c077a2997) clustering algorithm.

The implemention re-uses [this gist](https://gist.github.com/baali/7983261).

Example use:
```
import majorclust
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "foo blub baz",
    "foo bar baz",
    "asdf bsdf csdf",
    "foo bab blub",
    "csdf hddf kjtz",
    "123 456 890",
    "321 890 456 foo",
    "123 890 uiop",
]
mc = majorclust.MajorClust(sim_threshold=0.0)
X = TfidfVectorizer().fit_transform(texts)
mc.fit(X)

d = {}
for text, label in zip(texts, mc.labels_):
    d[label] = d.get(label, [])
    d[label].append(text)

for label, texts in sorted(d.items()):
    print("Cluster id %d:" % label)
    for t in texts:
        print(t)
    print("="*20)
```

Output:
```
Cluster id 1:
foo blub baz
foo bar baz
foo bab blub
====================
Cluster id 4:
asdf bsdf csdf
csdf hddf kjtz
====================
Cluster id 7:
123 456 890
321 890 456 foo
123 890 uiop
====================
```