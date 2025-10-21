---
title: 'Implementing a plugin for the Narwhals library: into the fray'

---

# Implementing a plugin for the Narwhals library: into the fray

I had the fortune to be accepted for Quansight's yearly open source internship, and during that time I worked on the [Narwhals](https://narwhals-dev.github.io/narwhals/) python library. I got to work with and meet some amazing people; for more background on that, see this post[TODO: INSERT LINK WHEN IT'S UP]. The present post is an extended version of the former; here we get an opportunity to dive more into the technical detail. 

**To start with, what is Narwhals?**

In a nutshell, it is a translation layer that allows you to write libraries for (mostly) Data Science tasks that will accept any number of dataframe types: pandas, Polars, PySpark, DuckDB to name but a few (the Narwhals landing page lists all the supported frameworks).  

It is the magical layer that makes it possible for you to pass both a pandas dataframe or a PySpark dataframe to [Plotly](https://plotly.com/blog/chart-smarter-not-harder-universal-dataframe-support/). 

# 1. Why might we need a plugin for Narwhals? 

Let's say you're already using a library which uses Narwhals, i.e. Plotly. Then you discover a new fancy dataframe library which you can't do without, but Narwhals doesn't support it yet. Wouldn't it be great if you could pass your dataframe natively to Plotly? 

By writing a Narwhals plugin, you'll be able to do just that, with zero required changes in either Plotly nor Narwhals.

But what do we mean by 'pass your dataframe natively'? Using daft in a simple example, you can see here that we're able to manipulate a daft dataframe from within Narwhals and perform computations on it. We're able to do this because we've written a plugin, [narwhals-daft](https://github.com/MarcoGorelli/narwhals-daft), for the [daft library](https://docs.daft.ai/en/stable/). 


```python
import daft

import narwhals as nw

daft_df = daft.from_pydict({
        "A": [1, 2, 3, 4, 5],
        "fruits": ["banana", "banana", "apple", "apple", "banana"],
        "B": [5, 4, 3, 2, 1],
    })

df = nw.from_native(daft_df)

print(f'Using the Narwhals `from_native` function on a daft dataframe, we obtain a {type(df)}. \n') 
print('Once we perform operations on it within Narwhals and collect it to show the result, we obtain a Narwhals DataFrame:')
print(df.with_columns(a_plus_1 = nw.col('A')+1).collect('polars')) 
```

```
Using the Narwhals `from_native` function on a daft dataframe, we obtain a <class 'narwhals.dataframe.LazyFrame'>. 

Once we perform operations on it within Narwhals and collect it to show the result, we obtain a Narwhals DataFrame:
┌─────────────────────────────────┐
|       Narwhals DataFrame        |
|---------------------------------|
|shape: (5, 4)                    |
|┌─────┬────────┬─────┬──────────┐|
|│ A   ┆ fruits ┆ B   ┆ a_plus_1 │|
|│ --- ┆ ---    ┆ --- ┆ ---      │|
|│ i64 ┆ str    ┆ i64 ┆ i64      │|
|╞═════╪════════╪═════╪══════════╡|
|│ 1   ┆ banana ┆ 5   ┆ 2        │|
|│ 2   ┆ banana ┆ 4   ┆ 3        │|
|│ 3   ┆ apple  ┆ 3   ┆ 4        │|
|│ 4   ┆ apple  ┆ 2   ┆ 5        │|
|│ 5   ┆ banana ┆ 1   ┆ 6        │|
|└─────┴────────┴─────┴──────────┘|
└─────────────────────────────────┘
```

As you can see, we've been able to pass a daft dataframe to Narwhals' `from_native` and get a Narwhals LazyFrame on which we can then do standard Narwhals operations. 

**This post is about what we had to do to make this happen (section 2), how you can go about making your own plugin (section3), and some of the considerations we had to think about along the way (4 & 5).** 


# 2. Making Narwhals pluggable: a proposed pattern

## 2.1 Prelude: what's a Protocol? 

Before discussing the code architecture, it is worth spending a moment on Protocols in Python, as these figure prominently in Narwhals. (A tutorial on Protocols is beyond the scope for this piece, but see [Python Protocols: Leveraging Structural Subtyping](https://realpython.com/python-protocol/) for a friendly introduction; additionally, this discussion helped me untangle abstract base classes from Protocols and helped clarify things for me: [Abstract Base Classes and Protocols: What Are They? When To Use Them?? Lets Find Out!](https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/))

In a nutshell, Protocols allow you to create abstract patterns that can work with different types of objects, as long as they have a common shape, or more correctly: interface. If you think of Narwhal’s capacity to work with multiple types of dataframe libraries, you can see why this would be very handy indeed: using Protocols, you can make sure that any class with the same methods and properties can be used in a context, even if classes do not inherit from the same base class. 

A toy example below shows how, because we have a Protocol for dataframes `FancyDataframeProtocol` which dictates the method greater than (Dunder `__gt__`) can take any type, we can have a function in a dataframe class that can return the max for both integers and strings. 

```python
# adapted from https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/

from typing import TypeVar, Protocol, Any

class FancyDataframeProtocol(Protocol):
    ...
    def __gt__(self, other: Any) -> bool:
        ...

T = TypeVar("T", bound=FancyDataframeProtocol)

class DataframeClassWithCustomMax():
    def max(self, x: T, y: T) -> T:
        if x > y:
            return x
        return y

find_max = DataframeClassWithCustomMax()
max_int = find_max.max(4, 5)
max_str = find_max.max("hello", "world!")

print(max_int)
print(max_str)
```

```
5
world!
```


## 2.2 The code architecture  

Narwhals already has guidelines on [how to extend the library](https://narwhals-dev.github.io/narwhals/extending/), although these are experimental and as yet untested. Nevertheless, they provide some must-haves which any plugin will eventually need to contain. 

We decided to initially implement the bare minimum, as this would show how a plugin could function in principle and get a discussion going with the community around design. 

**Our structure for narwhals-daft thus consists of:**

```
├── narwhals_daft
│   ├── __init__.py
│   ├── dataframe.py
│   ├── expr.py
│   ├── namespace.py
│   └── utils.py
├── pyproject.toml
├── README.md
```
### 2.2.1 a top-level `pyproject.toml` file with an `entry point` defined 

```
...

[project.entry-points.'narwhals.plugins']
narwhals-daft = 'narwhals_daft'
...
```

This information is used to create a connection with Narwhals (see section 3 for details) and every future plugin will need its own `pyproject.toml`, adapted to the particular library it is providing a plugin for. 


### 2.2.2 The top-level `__init__.py` file giving access to 3 crucial utilities:

Apart from the `.toml` file, this is where the connection to the Narwhals library happens. The file contains two functions, `__narwhals_namespace__` and `is_native` as well as a constant `NATIVE_PACKAGE` which gives the name of the package we're making a plugin for. 

The `__narwhals_namespace__` acts as the entry point to the library. Given the version of Narwhals, it returns a `DaftNamespace`, which can be wrapped around a non-narwhals dataframe (referred to as "native object" in the Narwhals terminology). The `DaftNamespace` makes a `from_native` function available, which allows the native object to be read into a compliant object (see sections 2.2.5 and 2.3), on which typical Narwhals operations can be carried out whilst still retaining the original data and data structure.

The `is_native` function simply checks if we are dealing with a daft dataframe. Note it is only at this point that we import the daft library, rather than when loading the plugin (see section 2.3 for further discussion of this aspect).  

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals_daft.namespace import DaftNamespace
    from narwhals_daft.dataframe import DaftLazyFrame

    from narwhals.utils import Version
    from typing_extensions import TypeIs


def __narwhals_namespace__(version: Version) -> DaftNamespace:  # noqa: N807
    from narwhals_daft.namespace import DaftNamespace

    return DaftNamespace(version=version)

def is_native(native_object:object) -> TypeIs[DaftLazyFrame]:
    import daft
    return isinstance(native_object, daft.DataFrame)


NATIVE_PACKAGE = "daft"
```

We can further see that in order for these functions to work, we'll need to implement three further classes: 

### 2.2.3 the DaftLazyFrame class

The top-level `is_native` function needs this; it lives inside the `dataframe.py` file. It contains mostly functions associated with dataframe operations. The `DaftLazyFrame` inherits from Narwhals' `CompliantLazyFrame`, which is a Protocol all plugins must implement. The methods prescribed by the `CompliantLazyFrame` need to be concretely implemented in `DaftLazyFrame`, even if only marked as `not_implemented`:

```python
...
from narwhals.typing import CompliantLazyFrame
...

class DaftLazyFrame(
    CompliantLazyFrame["DaftExpr", "daft.DataFrame", "LazyFrame[daft.DataFrame]"],
    ValidateBackendVersion,
):
    _implementation = Implementation.UNKNOWN

    ...

    @staticmethod
    def _is_native(obj: daft.DataFrame | Any) -> TypeIs[daft.DataFrame]:
        return isinstance(obj, daft.DataFrame)
    ...
    
    when: not_implemented = not_implemented()
    ...
```

### 2.2.4 the DaftExpr class

As we can see, the `DaftLazyFrame` needs an Expression class, in our case `DaftExpr` in the `expr.py` file. It houses methods that transform columns or series and are lazily evaluated. 

It inherits from the Narwhals' `LazyExpr`, which again contains a Protocol which stipulates all the methods any class inheriting from it will need to implement. 

```python
...
from narwhals._compliant import LazyExpr
...

class DaftExpr(LazyExpr["DaftLazyFrame", "Expression"]):
    _implementation = Implementation.UNKNOWN
...
    def sum(self) -> Self:
        def f(expr: Expression) -> Expression:
            return coalesce(expr.sum(), lit(0))

        return self._with_callable(f)

    def n_unique(self) -> Self:
        return self._with_callable(
            lambda _input: _input.count_distinct() + _input.is_null().bool_or()
        )
```
(For a good introduction to expressions, see [Expressions are coming to pandas!](https://labs.quansight.org/blog/pandas_expressions))

### 2.2.5 the DaftNamespace class

The class we return in the top-level `__narwhals_namespace__` function. It is defined in the `namespace.py` file and implements the interface defined by Narwhals' `LazyNamespace` Protocol. It defines methods we can apply to a lazy dataframe. 

```python
...
from narwhals._compliant.namespace import LazyNamespace
...

class DaftNamespace(LazyNamespace[DaftLazyFrame, DaftExpr, daft.DataFrame]):
    _implementation: Implementation = Implementation.UNKNOWN
    
    def __init__(self, *, version: Version) -> None:
        self._version = version

    def from_native(self, native_object: daft.DataFrame) -> DaftLazyFrame:
        return DaftLazyFrame(native_object, version=self._version)
    ...
    
```

## 2.3 How the plugin connects to Narwhals

**Note that plugin creators are not expected to modify the code within Narwhals, nevertheless it is useful to know how this works.**

In the Narwhals `translate.py` file, we read a native object and iterate over the modules which support the different dataframe types to try to read the native object. Once we've tried all the frameworks, we also check if we can read this object via a plugin (if this doesn't work either, we return an error message):

```python
# narwhals/translate.py
    ...
    compliant_object = plugins.from_native(native_object, version)
    if compliant_object is not None:
        return _translate_if_compliant(
            ...

    if not pass_through:
        msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(native_object)}"
        raise TypeError(msg)
    return native_object
```

As you can see, this calls the `from_native` function of the `plugins` module in Narwhals. The `plugins.py` file is where we store all our plugin-related utilities.

Since the plugin sits outside of Narwhals and should not alter its code, we have created a connection point within Narwhals, also known as contract. The contract uses the entry point defined in the section of the `pyproject.toml` file of the plugin. In `narwhals/plugins.py` we use this with the `importlib` library to detect installed plugins (see `_discover_entrypoints` function below). Note in the `plugins.py` file we only try to load the plugin in `_iter_from_native` once we've detected entry points. 

In that same function, `_is_native_plugin` calls two of the top-level utilities of the plugin: the package name (`plugin.NATIVE_PACKAGE`) and the `is_native` function. In order to be able to use these, we've had to define a Protocol for the Plugin `class Plugin(Protocol[FrameT, FromNativeR_co])` as all objects in Narwhals need to be supported by a Protocol. 

Still in `_iter_from_native`, the translation to a Narwhals-compliant dataframe happens via the namespace of the plugin, accessed via the plugin's top-level `__narwhals_namespace__` function, which allows the use of the plugin's namespace `from_native` function. Again we had to define a Protocol for the plugin's namespace to be able to use this in Narwhals (`class PluginNamespace(CompliantNamespace[FrameT, Any], Protocol[FrameT, FromNativeR_co])`).

```python
# narwhals/plugins.py
...

if TYPE_CHECKING:
    from importlib.metadata import EntryPoints
    ...
    
@cache
def _discover_entrypoints() -> EntryPoints:
    from importlib.metadata import entry_points as eps

    group = "narwhals.plugins"
    if sys.version_info < (3, 10):
        return cast("EntryPoints", eps().get(group, ()))
    return eps(group=group)


class PluginNamespace(CompliantNamespace[FrameT, Any], Protocol[FrameT, FromNativeR_co]):
    def from_native(self, data: Any, /) -> FromNativeR_co: ...


class Plugin(Protocol[FrameT, FromNativeR_co]):
    NATIVE_PACKAGE: LiteralString

    def __narwhals_namespace__(
        self, version: Version
    ) -> PluginNamespace[FrameT, FromNativeR_co]: ...
    def is_native(self, native_object: object, /) -> bool: ...

...


def _is_native_plugin(native_object: Any, plugin: Plugin) -> bool:
    pkg = plugin.NATIVE_PACKAGE
    return (
        sys.modules.get(pkg) is not None
        and _might_be(type(native_object), pkg)  # type: ignore[arg-type]
        and plugin.is_native(native_object)
    )


def _iter_from_native(native_object: Any, version: Version) -> Iterator[CompliantAny]:
    for entry_point in _discover_entrypoints():
        plugin: Plugin = entry_point.load()
        if _is_native_plugin(native_object, plugin):
            compliant_namespace = plugin.__narwhals_namespace__(version=version)
            yield compliant_namespace.from_native(native_object)

...

```

# 3. In summary, what users need to create their own plugin

Now that we understand the architecture of the plugin and have seen where the top-level utilities are called within Narwhals, we can see what a user would need to implement to make their own library accessible to Narwhals:

1. The entry point in the .toml file: `[project.entry-points.'narwhals.plugins']` must be part of every plugin. This line will be the same for all plugins, whereas the next line will need to be adapted to the name of the particular library the plugin is made for, namely:
    
        `narwhals-<insert libraryname> = 'narwhals_<insert libraryname>'`
2. The two top-level functions in `__init__.py`: `is_native object`,  `from_native` and the constant `NATIVE_PACKAGE` with the name of the package.
3. the 3 classes: `CompliantLazyFrame`, `Expression` & `Namespace` (in our daft-specific use case, `DaftLazyFrame`,`DaftExpr` & `DaftNamespace`). In section 2.2 we have explained how the top-level functions rely on these classes and we have shown in 2.3 how the top-level functions are used within the Narwhals library. Since plugin development is not expected to affect Narwhals development, following the structure outlined in those sections is advisable. Moreover, the methods prescribed by the Narwhals protocols for these classes will need to be implemented for the particular library. 

# 4. Issues we've considered when creating this solution

## 4.1 Finding the most lightweight approach to importing the framework of the plugin

Within the plugin, it was important to find a way of detecting whether the framework the plugin is written for is present, without going through the slow step of loading the framework itself. This is so that we can avoid importing `daft` if the user is not actually using a daft dataframe. for example, if a user has installed the (not-yet-existent) plugins `narwhals-xarray`, `narwhals-daft`, `narwhals-grizzlies`, then it should be possible to detect which plugin to use for a given user input without having to import all of `xarray`, `daft`, and `grizzlies`. If you recall the structure of the top-level `__init__.py` file (see 2.2.2), we are indeed only importing daft once we're checking a dataframe.

## 4.2 Ensuring backwards-compatibility

### 4.2.1 Related materials: the case of plugins in scikit-learn

It may be instructive to consider the issues faced by other packages when planning our own plugin. 

For scikit-learn, it is a known issue that even with minor releases, extensions currently break (true as of 01/09/25, see discussion [here](https://github.com/scikit-learn/scikit-learn/issues/31912)). 

There is currently a debate about how to fix this, with one proposal suggesting a more abstracted architecture where potentially breaking changes can be made safe by being placed between stable public-facing layers. On the other hand, this represents a quite significant rewrite of the package which may not sit well with users accustomed to more procedural scientific code. Scikit-learn is an established package with a long history and is widely used; this shows that questions around interfaces to other packages should be considered as early on as possible.

### 4.2.2 Narwhals

Narwhals has a strong policy of [backwards compatibility](https://narwhals-dev.github.io/narwhals/backcompat/), mainly enforced through the use of a `stable` namespace. However, that promise does not extend to its internal Compliant protocols.

Given that plugin authors use the Compliant Protocols, we need to ensure that these protocols are stable enough that users' plugins don't break with every minor Narwhals release

**Here are some guidelines which we (Narwhals developers) aim to follow, to minimise the potential disruption on plugins authors.**

> If an existing Protocol method is updated to accept additional arguments:
> 
> - New parameter(s) MUST be introduced with a default value
> - The behavior of the existing signature MUST not change
> - The Narwhals-level SHOULD handle the default case, but all other cases remain undefined

We may need to deviate from them if strictly necessary, but we hope that this will be rare.

**The advice to plugin authors to avoid breakage with new Narwhals methods is therefore:**

> - Only use the public methods from the compliant protocols. 
> - Don't rely on anything starting with an underscore

# 5. Remaining issues

## 5.1 Should the typing be able to deal with unrecognised but compliant dataframes? 

Should we be able to work with a dataframe, even if it is not associated with a particular backend? At the moment, if we have narwhals-daft installed, we should be able to do the following as daft has the ability to create a dataframe from a dictionary via daft.from_pydict. 


```python
import daft
import narwhals as nw

df = nw.from_native({'a': [1,2,3]})
reveal_type(df)
```

However, this fails because the dictionary is not recognised as a native dataframe by Narwhals' typing. Should this change and if so, how?


## 5.2 How should plugins/extensions interact with `Implementation.UNKNOWN`?

In Narwhals, certain functions depend on the implementation of the backend being passed, for example `scan_parquet` & `from_dict`. Thus if someone has the daft-plugin installed, they should be able to do:

```
df: nw.LazyFrame # `df` is a narwhals LazyFrame backed by Daft

nw.scan_parquet(file, backend=df.implementation)

# as wel as:

nw.scan_parquet(file, backend='narwhals-daft')
```

This has prompted an ongoing rethink of the use of `Implementation.UNKNOWN`. 

