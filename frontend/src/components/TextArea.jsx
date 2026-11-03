import { MAX_TEXT_LENGTH } from "../api";

/** Textarea that submits on Ctrl/Cmd + Enter. */
export default function TextArea({ onSubmit, ...props }) {
  function handleKeyDown(event) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      onSubmit();
    }
  }

  return <textarea className="textarea" maxLength={MAX_TEXT_LENGTH} onKeyDown={handleKeyDown} {...props} />;
}
