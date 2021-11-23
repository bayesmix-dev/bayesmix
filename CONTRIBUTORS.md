# Contributors of `bayesmix`

## Pull Requests

All contributions are subject to code review via a pull requests (PR) from a fork of the repository.
A continuous integration tool will automatically run the unit tests, which must pass in order to get an approval of the PR.
Since the continuous integration tool can run for a maximum of 3000 minutes per month, we cannot afford to run unit tests at every commit. So follow these rules

1- If you want early feedback on a PR, open it as a _draft pull request_ (tests will not run)

2- Avoid pushing to a remote branch at every commit when a PR is open, but only when stuff is really done

3- If a PR takes several rounds of review, convert the PR to draft between these iterations

4- If the tests don't pass, fix the tests on your local machine before pushing again

## Style

This library follows the official [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
This includes using the `.cc` extension for source files and `.h` for header files, not `.cpp` or `.hpp`.

The recommended way to adhere to the style is to install the `clang-format` package to allow automatic formatting adjustment of any file, for instance by running:

```shell
sudo apt-get install clang-format
```

Then, install the pre-commit hooks by running

```shell
./bash/setup_pre_commit.sh
```

When committing, the hooks will format the file for you, clear eventual Jupyter notebooks and perform other checks.

Alternatively, you can use manually the package to format files appropriately with:

```shell
clang-format -i --style=file yourfile1.h yourfile2.cc
```

which reads the appropriate settings from the `.clang-format` file at the root folder.
You can also use wildcards to include multiple files at once.
For example, the following command will read all files in the `src/hierarchies` subfolder which end in `.h`:

```shell
clang-format -i --style=file src/hierarchies/*.h
```

## Future steps

- Extension to normalized random measures
- Using HMC / MALA MCMC algorithm to sample from the cluster-specific full conditionals when it's not conjugate to the base measure
- R package
- Please check out the [issue page](https://github.com/bayesmix-dev/bayesmix/issues) for more planned enhancements.

## A note on Hierarchies and Mixings

This library implements `Hierarchy` and `Mixing` objects through the [Curiously Recurring Template Pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).
Please check out [src/hierarchies/README.md](src/hierarchies/README.md) for more details.
