# Add a CoreML model into a Swift package

Swift packages are a nice way to distribut a piece of code.
It is possible to add a CoreML model into a Swift package.
But it still requires a bit of code.


## Configuration

The following procedure has been tested on the following configuration:

- MacBook Pro (16-inch, 2019)
- macOS Catalina 10.15.7
- XCode 12.1


## Procedure

### Convert the MLModel

The MLModel cannot be used directly in the Swift package. It first needs to be converted.

```
$ cd /path/to/folder/containg/mlmodel
$ xcrun coremlcompiler compile MyModel.mlmodel .
$ xcrun coremlcompiler generate MyModel.mlmodel . --language Swift
```

The first `xcrun` command will compile the model and create a folder named `MyModel.mlmodelc`.
The second `xcrun` command will generate a `MyModel.swift` file.

### Add the model to the Swift package

We consider that a Swift package already exists and is located in `/path/to/MyPackage/`.

1. Copy the `MyModel.mlmodelc` folder and `MyModel.swift` file into the folder `/path/to/MyPackage/Sources/MyPackage`
2. Modify the file `Package.swift` to add the `MyModel.mlmodelc` in the target resources: 

```
.target(
    name: "MyPackage",
    dependencies: [],
        resources: [.process("MyModel.mlmodelc")]),
```

### Instantiate MyModel

In the Swift code, simply create an instance of MyModel:

```
let model = try? MyModel(configuration: MLModelConfiguration())
```

or:

```
let url = Bundle.module.url(forResource: "MyModel", withExtension: "mlmodelc")!
let model = try? MyModel(contentsOf: url, configuration: MLModelConfiguration())
```


## Troubleshooting

I got a `Type 'MLModel' has no member '__loadContents'` error at first.
This seems to be a bg related to XCode 12.
I simply commented the 2 functions that caused a problem.

See [here](https://stackoverflow.com/questions/63917164/ml-build-error-for-catalyst-xcode-12-gm) and [here](https://github.com/apple/coremltools/issues/930) for more information.