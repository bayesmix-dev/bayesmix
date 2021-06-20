# Contributors of bayesmix
## Recommendations
Before commiting anything, please install the pre-commit hooks by running
```shell
./bash/setup_pre_commit.sh
```
This automatically clears the output of all Jupyter notebooks, if necessary.

## Style
This library follows the official [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
You can install the `clang-format` package to allow automatic formatting adjustment of any file.
You can do so by running
```shell
clang-format -i --style=file yourfile.cc
```
which reads the appropriate settings from the ```.clang-format``` file at the root folder.

## Future steps
* Extension to normalized random measures
* Using HMC / MALA MCMC algorithm to sample from the cluster-specific full conditionals when it's not conjugate to the base measure
* R package
* Please check out the [issue page](https://github.com/bayesmix-dev/bayesmix/issues) for more planned enhancements.

## Hierarchies
This library implements hierarchy objects through the [Curiously Recurring Template Pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).
Please check out [src/hierarchies/README.md](src/hierarchies/README.md) for more details.
