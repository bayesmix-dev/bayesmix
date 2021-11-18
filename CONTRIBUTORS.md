# Contributors of `bayesmix`
## Recommendations
Before commiting anything, please install the pre-commit hooks by running
```shell
./bash/setup_pre_commit.sh
```
This automatically clears the output of all Jupyter notebooks, if necessary.


## Style
This library follows the official [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
This includes using the `.cc` extension for source files and `.h` for header files, not `.cpp` or `.hpp`.

You can install the `clang-format` package to allow automatic formatting adjustment of any file, for instance by running:
```shell
sudo apt-get install clang-format
```
Finally, you can use the package to format files appropriately with:
```shell
clang-format -i --style=file yourfile1.h yourfile2.cc
```
which reads the appropriate settings from the ```.clang-format``` file at the root folder.
You can also use wildcards to include multiple files at once.
For example, the following command will read all files in the `src/hierarchies` subfolder which end in `.h`:
```shell
clang-format -i --style=file src/hierarchies/*.h
```


## Future steps
* Extension to normalized random measures
* Using HMC / MALA MCMC algorithm to sample from the cluster-specific full conditionals when it's not conjugate to the base measure
* R package
* Please check out the [issue page](https://github.com/bayesmix-dev/bayesmix/issues) for more planned enhancements.


## A note on Hierarchies and Mixings
This library implements `Hierarchy` and `Mixing` objects through the [Curiously Recurring Template Pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).
Please check out [src/hierarchies/README.md](src/hierarchies/README.md) for more details.
