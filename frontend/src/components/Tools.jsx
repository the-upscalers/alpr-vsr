import { FaUndo, FaRedo, FaTrash, FaGithub, FaNpm } from "react-icons/fa";

export default function Tools({ annotationRef }) {
	return (
		<>
			{/* Branding */}
			<div className="text-white font-bold text-lg">
				<span>ALPR-VSR</span>
			</div>

			{/* Toolbar Buttons */}
			<div className="flex items-center gap-4">
				{/* Undo Button */}
				<button
					onClick={annotationRef.current?.undo}
					title="Undo or CTRL+Z"
					className="text-white bg-stone-700 hover:bg-stone-600 p-2 rounded-full transition"
				>
					<FaUndo size={18} />
				</button>
				{/* Redo Button */}
				<button
					onClick={annotationRef.current?.redo}
					title="Redo or CTRL+Y"
					className="text-white bg-stone-700 hover:bg-stone-600 p-2 rounded-full transition"
				>
					<FaRedo size={18} />
				</button>
				{/* Delete Button */}
				<button
					onClick={annotationRef.current?.deleteShape}
					title="Delete or delete key"
					className="text-white bg-red-700 hover:bg-red-600 p-2 rounded-full transition"
				>
					<FaTrash size={18} />
				</button>
			</div>
		</>
	);
}
