import { useRef, useState } from "react";
import AnnotationTools from "./components/AnnotationTools";
import Settings from "./components/Settings";
import Tools from "./components/Tools";
import { FileUp, Settings as SettingsIcon } from "lucide-react";
import { TwoDVideoAnnotation } from "react-video-annotation-tool";

const videoControls = {
	// autoPlay: true,
	// loop: true
};

function App() {
	const [showSettings, setShowSettings] = useState(false);
	const [selectedTool, setSelectedTool] = useState(null);
	const [allAnnotations, setAllAnnotations] = useState([]);
	const [annotationColor, setAnnotationColor] = useState("red");
	const [annotationData, setAnnotationData] = useState(null);
	const [videoUrl, setVideoUrl] = useState("");

	const annotationRef = useRef(null);
	const fileInputRef = useRef(null);

	const handleSelectAnnotationData = (data) => {
		setAnnotationData(data);
	};

	const handleFileUpload = (event) => {
		const file = event.target.files?.[0];
		if (file) {
			const url = URL.createObjectURL(file);
			setVideoUrl(url);
			// Reset annotations when new video is uploaded
			setAllAnnotations([]);
			setAnnotationData(null);
		}
	};

	const triggerFileUpload = () => {
		fileInputRef.current?.click();
	};

	console.log(annotationData);

	return (
		<div className="w-screen h-screen bg-stone-900 overflow-hidden flex flex-col">
			{/* Tools */}
			<div className="rounded-full mt-2 py-2 flex items-center justify-between w-[95%] mx-auto h-14 bg-stone-800 px-4 shadow-md">
				<Tools annotationRef={annotationRef} />
				<div className="flex items-center gap-2">
					<input
						type="file"
						ref={fileInputRef}
						onChange={handleFileUpload}
						accept="video/*"
						className="hidden"
					/>
					<button
						onClick={triggerFileUpload}
						className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-full transition"
					>
						<FileUp className="w-4 h-4" />
						<span>Upload Video</span>
					</button>
				</div>
			</div>

			<div className="flex flex-row gap-3 p-4 overflow-y-auto">
				{/* Video Player */}
				<div className="bg-stone-900 flex-1 rounded-3xl p-2">
					<div className="w-[90%] mx-auto">
						<TwoDVideoAnnotation
							rootRef={annotationRef}
							shapes={allAnnotations}
							setShapes={setAllAnnotations}
							videoUrl={videoUrl}
							selectedShapeTool={selectedTool}
							hideAnnotations={false}
							annotationColor={annotationColor}
							videoControls={videoControls}
							lockEdit={false}
							initialAnnotationData={annotationData}
							selectedAnnotationData={(data) =>
								handleSelectAnnotationData(data)
							}
						/>
					</div>
				</div>

				{/* Annotation Tools */}
				<div
					className="relative bg-stone-800 w-[400px] rounded-3xl shadow-lg m-2 pt-4 px-4 overflow-y-auto"
					style={{ scrollbarWidth: "none" }}
				>
					<div className="absolute top-4 right-4">
						<button
							onClick={() => setShowSettings(!showSettings)}
							className={`text-white ${
								showSettings ? "bg-blue-600" : "bg-stone-700"
							} hover:bg-blue-600 p-1 rounded-full transition`}
						>
							<SettingsIcon className="w-4 h-4" />
						</button>
					</div>
					{showSettings ? (
						<Settings />
					) : (
						<>
							<h2 className="text-white text-center text-xl font-semibold mb-4">
								Annotation Tools
							</h2>
							<AnnotationTools
								selectedTool={selectedTool}
								setSelectedTool={setSelectedTool}
								annotationColor={annotationColor}
								setAnnotationColor={setAnnotationColor}
							/>
						</>
					)}
				</div>
			</div>
		</div>
	);
}

export default App;
