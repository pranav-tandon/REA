import { useState } from "react";

export default function Home() {
  const [userQuery, setUserQuery] = useState("");
  const [chatResponse, setChatResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_query: userQuery }),
      });
      const data = await res.json();
      setChatResponse(data.response);
    } catch (error) {
      console.error(error);
      setChatResponse("Error communicating with backend.");
    }
    setLoading(false);
  };

  return (
    <main style={{ padding: "1rem" }}>
      <h1>REA - Buyer Chat</h1>
      <textarea
        rows={3}
        style={{ width: "100%", marginBottom: "0.5rem" }}
        value={userQuery}
        onChange={(e) => setUserQuery(e.target.value)}
        placeholder="Ask about properties..."
      />
      <div>
        <button onClick={handleSend} disabled={loading}>
          {loading ? "Loading..." : "Send"}
        </button>
      </div>
      {chatResponse && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Bot Response:</h3>
          <p>{chatResponse}</p>
        </div>
      )}
    </main>
  );
}
