# ModifiedHausdorffDistance

![GitHub Actions](https://github.com/TuuPu/ModifiedHausdorffDistance/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/TuuPu/ModifiedHausdorffDistance/branch/main/graph/badge.svg?token=9YFEKBEPSE)](https://codecov.io/gh/TuuPu/ModifiedHausdorffDistance)



## Documentation

[Requirement documentation](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/requirement_spec.md)

[Testing documentation](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/testing_document.md)

### Weekly reports

[Week 1 report](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/weekly_report_1.md)

[Week 2 report](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/weekly_report_2.md)

[Week 3 report](https://github.com/TuuPu/ModifiedHausdorffDistance/blob/main/documentation/weekly_report_3.md)

### Get the project running

`poetry install`

### Testing

```
poetry shell
pytest
```

### Coverage

```
poetry shell
coverage run --branch -m pytest; coverage html
```

This creates a htmlcov directory and by opening the html.index file you can see the coverage report. You can access the same information by going to codecov from the repository front page by clicking the badge.

### Running the program

```
poetry shell
python src/app.py
```

### Pylint

```
poetry shell
pylint src
```
