import AppKit
import CoreImage
import Foundation
import Vision

enum RemoveBackgroundError: Error {
	case usage
	case loadImage
	case noMask
	case maskGeneration
	case renderFailure
	case pngEncoding
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
	fputs("usage: remove_background <input> <output>\n", stderr)
	throw RemoveBackgroundError.usage
}

let inputURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])

guard
	let imageSource = CGImageSourceCreateWithURL(inputURL as CFURL, nil),
	let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)
else {
	throw RemoveBackgroundError.loadImage
}

let request = VNGenerateForegroundInstanceMaskRequest()
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])

guard let observation = request.results?.first else {
	throw RemoveBackgroundError.noMask
}

let maskBuffer = try observation.generateScaledMaskForImage(forInstances: observation.allInstances, from: handler)
let inputImage = CIImage(cgImage: cgImage)
let maskImage = CIImage(cvPixelBuffer: maskBuffer)
let transparentBackground = CIImage(color: .clear).cropped(to: inputImage.extent)

guard let filter = CIFilter(name: "CIBlendWithMask") else {
	throw RemoveBackgroundError.renderFailure
}
filter.setValue(inputImage, forKey: kCIInputImageKey)
filter.setValue(transparentBackground, forKey: kCIInputBackgroundImageKey)
filter.setValue(maskImage, forKey: kCIInputMaskImageKey)

guard
	let outputImage = filter.outputImage,
	let rendered = CIContext().createCGImage(outputImage, from: inputImage.extent)
else {
	throw RemoveBackgroundError.renderFailure
}

let bitmap = NSBitmapImageRep(cgImage: rendered)
guard let pngData = bitmap.representation(using: NSBitmapImageRep.FileType.png, properties: [:]) else {
	throw RemoveBackgroundError.pngEncoding
}

try pngData.write(to: outputURL)
