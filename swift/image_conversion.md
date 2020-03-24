# Image conversion

Image conversion is not easy. I'm gathering bellow some conversions I was able to put together. They might not be the right way to converting images, sorry if it's the case. But at least that's a start and hopefully it will help some of you.


# Conversion UIImage / CGImage

### UIImage to CGImage

```swift
let myCGImage = myUIImage.cgImage
```

### CGImage to UIImage

```swift
let myUIImage = UIImage(cgImage: myCGOutput)
```


# Conversion UIImage / CIImage

### UIImage to CIImage

```swift
let myCIImage = CIImage(image: myUIImage)
```


### CIImage to UIImage

```swift
// We assume we know the image size
let size = CGSize(width: width, height: height)

// CIImage to UIImage conversion
UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
UIImage(ciImage: myCIImage).draw(at: CGPoint(x: 0, y: 0))
let myUIImage = UIGraphicsGetImageFromCurrentImageContext()
UIGraphicsEndImageContext()
```


# Conversion CGImage / CIImage

### CGImage to CIImage

```swift
let myCIImage = CIImage(cgImage: myCGImage)
```

### CIImage to CGImage

```swift
// We assume we know the image size
let context = CIContext(options: nil)
let myCGImage = context.createCGImage(myCIImage, from: CGRect(x: 0, y: 0, width: width, height: height))
```


# Conversion CGImage / CVPixelBuffer

### CGImage to CVPixelBuffer

This conversion was the most difficult one to write.

```swift
func conversion(myCGImage: CGImage?) -> CVPixelBuffer? {

    guard let myCGImage = myCGImage else {
        return nil
    }
    
    // Attributes needed to create the CVPixelBuffer
    let attributes = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                      kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]

    // Create the input CVPixelBuffer
    var myCVPixelBuffer: CVPixelBuffer? = nil
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     myCGImage.width,
                                     myCGImage.height,
                                     kCVPixelFormatType_32ARGB,
                                     attributes as CFDictionary,
                                     &myCVPixelBuffer)

    // Status check
    if status != kCVReturnSuccess {
        return nil
    }

    // Fill the input CVPixelBuffer with the content of the input CGImage
    CVPixelBufferLockBaseAddress(myCVPixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(myCVPixelBuffer!),
                                  width: myCGImage.width,
                                  height: myCGImage.height,
                                  bitsPerComponent: myCGImage.bitsPerComponent,
                                  bytesPerRow: myCGImage.bytesPerRow,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            return nil
    }
    context.draw(myCGImage, in: CGRect(x: 0, y: 0, width: myCGImage.width, height: myCGImage.height))
    CVPixelBufferUnlockBaseAddress(myCVPixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

    return myCVPixelBuffer
}
```

### CVPixelBuffer to CGImage

```swift
// We assume we know the image size
let myCIImage = CIImage(cvPixelBuffer: myCVPixelBuffer)
let context = CIContext(options: nil)
let myCGImage = context.createCGImage(myCIImage, from: CGRect(x: 0, y: 0, width: width, height: height))

```