from visual_chatgpt.tools.blipvqa import BLIPVQA
from visual_chatgpt.tools.depth2image import Depth2Image
from visual_chatgpt.tools.hed2image import Hed2Image
from visual_chatgpt.tools.image2depth import Image2Depth
from visual_chatgpt.tools.image2hed import Image2Hed
from visual_chatgpt.tools.image2normal import Image2Normal
from visual_chatgpt.tools.image2pose import Image2Pose
from visual_chatgpt.tools.image2seg import Image2Seg
from visual_chatgpt.tools.image_captioning import ImageCaptioning
from visual_chatgpt.tools.image_editing import ImageEditing
from visual_chatgpt.tools.line2image import Line2Image
from visual_chatgpt.tools.normal2image import Normal2Image
from visual_chatgpt.tools.pose2image import Pose2Image
from visual_chatgpt.tools.scribble2image import Scribble2Image
from visual_chatgpt.tools.seg2image import Seg2Image
from visual_chatgpt.tools.t2i import T2I
from visual_chatgpt.tools.image2canny import Image2Canny
from visual_chatgpt.tools.image2scribble import Image2Scribble
from visual_chatgpt.tools.canny2image import Canny2Image
from visual_chatgpt.tools.pix2pix import Pix2Pix
from visual_chatgpt.tools.image2line import Image2Line
import os
from langchain.agents.tools import Tool


class ToolManager:
    def __init__(self) -> None:
        self.edit = (
            ImageEditing(device=os.environ["TOOL_EDIT"])
            if os.getenv("TOOL_EDIT")
            else None
        )
        self.i2t = (
            ImageCaptioning(device=os.environ["TOOL_I2T"])
            if os.getenv("TOOL_I2T")
            else None
        )
        self.t2i = T2I(device=os.environ["TOOL_T2I"]) if os.getenv("TOOL_T2I") else None
        self.image2canny = Image2Canny() if os.getenv("TOOL_IMAGE2CANNY") else None
        self.canny2image = (
            Canny2Image(device=os.environ["TOOL_CANNY2IMAGE"])
            if os.getenv("TOOL_CANNY2IMAGE")
            else None
        )
        self.image2line = Image2Line() if os.getenv("TOOL_IMAGE2LINE") else None
        self.line2image = (
            Line2Image(device=os.environ["TOOL_LINE2IMAGE"])
            if os.getenv("TOOL_LINE2IMAGE")
            else None
        )
        self.image2hed = Image2Hed() if os.environ["TOOL_IMAGE2HED"] else None
        self.hed2image = (
            Hed2Image(device=os.environ["TOOL_HED2IMAGE"])
            if os.getenv("TOOL_HED2IMAGE")
            else None
        )
        self.image2scribble = (
            Image2Scribble() if os.getenv("TOOL_IMAGE2SCRIBBLE") else None
        )
        self.scribble2image = (
            Scribble2Image(device=os.environ["TOOL_SCRIBBLE2IMAGE"])
            if os.getenv("TOOL_SCRIBBLE2IMAGE")
            else None
        )
        self.image2pose = Image2Pose() if os.getenv("TOOL_IMAGE2POSE") else None
        self.pose2image = (
            Pose2Image(device=os.environ["TOOL_POSE2IMAGE"])
            if os.getenv("TOOL_POSE2IMAGE")
            else None
        )
        self.BLIPVQA = (
            BLIPVQA(device=os.environ["TOOL_BLIPVQA"])
            if os.getenv("TOOL_BLIPVQA")
            else None
        )
        self.image2seg = Image2Seg() if os.getenv("TOOL_IMAGE2SEG") else None
        self.seg2image = (
            Seg2Image(device=os.environ["TOOL_SEG2IMAGE"])
            if os.getenv("TOOL_SEG2IMAGE")
            else None
        )
        self.image2depth = Image2Depth() if os.getenv("TOOL_IMAGE2DEPTH") else None
        self.depth2image = (
            Depth2Image(device=os.environ["TOOL_DEPTH2IMAGE"])
            if os.getenv("TOOL_DEPTH2IMAGE")
            else None
        )
        self.image2normal = Image2Normal() if os.getenv("TOOL_IMAGE2NORMAL") else None
        self.normal2image = (
            Normal2Image(device=os.environ["TOOL_NORMAL2IMAGE"])
            if os.getenv("TOOL_NORMAL2IMAGE")
            else None
        )
        self.pix2pix = (
            Pix2Pix(device=os.environ["TOOL_PIX2PIX"])
            if os.getenv("TOOL_PIX2PIX")
            else None
        )

    def get_tools(self):
        result = []

        if self.i2t:
            result.append(
                Tool(
                    name="Get Photo Description",
                    func=self.i2t.inference,
                    description="useful when you want to know what is inside the photo. receives image_path as input. "
                    "The input to this tool should be a string, representing the image_path. ",
                )
            )
        if self.t2i:
            result.append(
                Tool(
                    name="Generate Image From User Input Text",
                    func=self.t2i.inference,
                    description="useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                    "The input to this tool should be a string, representing the text used to generate image. ",
                )
            )
        if self.edit:
            result.append(
                Tool(
                    name="Remove Something From The Photo",
                    func=self.edit.remove_part_of_image,
                    description="useful when you want to remove and object or something from the photo from its description or location. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the object need to be removed. ",
                )
            )
            result.append(
                Tool(
                    name="Replace Something From The Photo",
                    func=self.edit.replace_part_of_image,
                    description="useful when you want to replace an object from the object description or location with another object from its description. "
                    "The input to this tool should be a comma seperated string of three, representing the image_path, the object to be replaced, the object to be replaced with ",
                )
            )

        if self.pix2pix:
            result.append(
                Tool(
                    name="Instruct Image Using Text",
                    func=self.pix2pix.inference,
                    description="useful when you want to the style of the image to be like the text. like: make it look like a painting. or make it like a robot. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the text. ",
                )
            )

        if self.BLIPVQA:
            result.append(
                Tool(
                    name="Answer Question About The Image",
                    func=self.BLIPVQA.get_answer_from_question_and_image,
                    description="useful when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the question",
                )
            )

        if self.image2canny:
            result.append(
                Tool(
                    name="Edge Detection On Image",
                    func=self.image2canny.inference,
                    description="useful when you want to detect the edge of the image. like: detect the edges of this image, or canny detection on image, or peform edge detection on this image, or detect the canny image of this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.canny2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Canny Image",
                    func=self.canny2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and a canny image. like: generate a real image of a object or something from this canny image, or generate a new real image of a object or something from this edge image. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description. ",
                )
            )
        if self.image2line:
            result.append(
                Tool(
                    name="Line Detection On Image",
                    func=self.image2line.inference,
                    description="useful when you want to detect the straight line of the image. like: detect the straight lines of this image, or straight line detection on image, or peform straight line detection on this image, or detect the straight line image of this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.line2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Line Image",
                    func=self.line2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and a straight line image. like: generate a real image of a object or something from this straight line image, or generate a new real image of a object or something from this straight lines. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description. ",
                )
            )
        if self.image2hed:
            result.append(
                Tool(
                    name="Hed Detection On Image",
                    func=self.image2hed.inference,
                    description="useful when you want to detect the soft hed boundary of the image. like: detect the soft hed boundary of this image, or hed boundary detection on image, or peform hed boundary detection on this image, or detect soft hed boundary image of this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.hed2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Soft Hed Boundary Image",
                    func=self.hed2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and a soft hed boundary image. like: generate a real image of a object or something from this soft hed boundary image, or generate a new real image of a object or something from this hed boundary. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        if self.image2seg:
            result.append(
                Tool(
                    name="Segmentation On Image",
                    func=self.image2seg.inference,
                    description="useful when you want to detect segmentations of the image. like: segment this image, or generate segmentations on this image, or peform segmentation on this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.seg2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Segmentations",
                    func=self.seg2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and segmentations. like: generate a real image of a object or something from this segmentation image, or generate a new real image of a object or something from these segmentations. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        if self.image2depth:
            result.append(
                Tool(
                    name="Predict Depth On Image",
                    func=self.image2depth.inference,
                    description="useful when you want to detect depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.depth2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Depth",
                    func=self.depth2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and depth image. like: generate a real image of a object or something from this depth image, or generate a new real image of a object or something from the depth map. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        if self.image2normal:
            result.append(
                Tool(
                    name="Predict Normal Map On Image",
                    func=self.image2normal.inference,
                    description="useful when you want to detect norm map of the image. like: generate normal map from this image, or predict normal map of this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.normal2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Normal Map",
                    func=self.normal2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and normal map. like: generate a real image of a object or something from this normal map, or generate a new real image of a object or something from the normal map. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        if self.image2scribble:
            result.append(
                Tool(
                    name="Sketch Detection On Image",
                    func=self.image2scribble.inference,
                    description="useful when you want to generate a scribble of the image. like: generate a scribble of this image, or generate a sketch from this image, detect the sketch from this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.scribble2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Sketch Image",
                    func=self.scribble2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and a scribble image or a sketch image. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        if self.image2pose:
            result.append(
                Tool(
                    name="Pose Detection On Image",
                    func=self.image2pose.inference,
                    description="useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. "
                    "The input to this tool should be a string, representing the image_path",
                )
            )
        if self.pose2image:
            result.append(
                Tool(
                    name="Generate Image Condition On Pose Image",
                    func=self.pose2image.inference,
                    description="useful when you want to generate a new real image from both the user desciption and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. "
                    "The input to this tool should be a comma seperated string of two, representing the image_path and the user description",
                )
            )
        return result