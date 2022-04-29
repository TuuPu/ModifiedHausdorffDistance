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

This creates a htmlcov directory and by opening the html.index file you can see the coverage report. You can access the same information by going to codecov from the 
repository $

### Running the program

Note that running the program runs it with pre-gathered data. The date of the data is stated in app.py and the values have been gathered by running the program on my 
computer. If you wish, you can run the whole program by commenting the data away and running the performance tests. I have noticed some differences in calculation times 
when my friends have tested my program. So it might be interesting to run the tests. Though running the tests takes close to an hour. So be prepared.

```
poetry shell
python src/app.py
```

### Pylint

```
poetry shell
pylint src  
```   
   
