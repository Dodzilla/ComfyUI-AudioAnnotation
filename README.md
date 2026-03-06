# ComfyUI Audio Annotation

Custom ComfyUI nodes for building audio-to-annotation workflows that emit downloadable JSON files.

Included nodes:

- `KTMAudioReferenceAnnotation`
  - Canonicalizes input audio to WAV
  - Runs `faster-whisper`
  - Anchors output to optional reference lyrics
  - Produces annotation JSON as a string
- `KTMSaveAudioAnnotationJson`
  - Writes the JSON string to `ComfyUI/output`
  - Returns the saved relative path so external job systems can fetch it

The intended deployment target is a dependency-managed Furgen/KillaTamata ComfyUI worker.
