# Markdown reminders


## Web links

* [github markdown basics](https://help.github.com/articles/markdown-basics/)

## Basics

* list 1
  * list 2
  * *italic*
* **bold**
* `verbatim monospace`

```
this is a
block containing
some code
```
```python
python = code
```

```scala
val scala = code
```


## Table

First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell


## Convert to HTML, code extraction, etc.

```bash
# convert to HTML
pandoc --from markdown_github --to html --standalone README.md --output README.html

# extract code blocks
## all code
cat MyFile.md | sed -n '/^```/,/^```/ p' | sed '/^```/ d' > MyFile.txt
## just bash
cat MyFile.md | sed -n '/^```bash/,/^```/ p' | sed '/^```/ d' > MyFile.sh
## leave whitespace between blocks
cat MyFile.md | sed -n '/^```bash/,/^```/ p' | sed 's/^```.*//g' > MyFile.sh

```



