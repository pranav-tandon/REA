import { useRouter } from "next/router";
import { useEffect, useState } from "react";

export default function FinalPage() {
  const router = useRouter();
  const { profileId } = router.query as { profileId?: string };

  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<string[]>([]);

  useEffect(() => {
    if (profileId) {
      finalizeSearch(profileId);
    }
  }, [profileId]);

  const finalizeSearch = async (pid: string) => {
    try {
      setLoading(true);
      setError("");
      // POST to /flow/finalize-search
      const res = await fetch("/api/flowProxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: "/flow/finalize-search",
          method: "POST",
          payload: { profileId: pid }
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to finalize search");
      }
      setRecommendations(data.top_recommendations || []);
    } catch (err: any) {
      console.error("Error finalizing search:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatInput.trim()) return;
    // Capture the user message and send it to your /chat endpoint.
    const userMsg = chatInput;
    setChatInput("");
    setChatHistory([...chatHistory, `User: ${userMsg}`]);
    try {
      const res = await fetch("/api/houseSearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: userMsg,
          type: "chat",
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Chat error");
      }
      setChatHistory((prev) => [...prev, `Assistant: ${data.response}`]);
    } catch (err: any) {
      console.error("Error in chat:", err);
      setChatHistory((prev) => [...prev, `Assistant: [Error: ${err.message}]`]);
    }
  };

  if (!profileId) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p>No profileId found.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p>Generating your final recommendations...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Top Recommendations</h1>
        {error && <p className="text-red-500 mb-4">{error}</p>}

        {recommendations.length === 0 && (
          <p>No recommendations found. Try adjusting your constraints.</p>
        )}

        <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
          {recommendations.map((rec, idx) => (
            <div key={idx} className="bg-gray-800 p-4 rounded shadow">
              <p className="font-bold">{rec.address}</p>
              <p className="text-sm text-gray-400">
                Price: ${rec.price?.toLocaleString?.() || rec.price}
              </p>
              <p className="mt-2 text-gray-200">{rec.justification}</p>
            </div>
          ))}
        </div>

        {/* Chat Box */}
        <div className="mt-8 bg-gray-800 p-4 rounded shadow">
          <h2 className="text-xl font-bold mb-2">Chat / Q&A</h2>
          <div className="space-y-2 mb-4 h-40 overflow-y-auto bg-gray-700 p-3 rounded">
            {chatHistory.map((line, i) => (
              <p key={i} className="text-sm">
                {line}
              </p>
            ))}
          </div>

          <div className="flex space-x-2">
            <input
              type="text"
              className="flex-1 p-2 bg-gray-600 rounded"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Ask a question about these listings..."
            />
            <button
              onClick={handleChat}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold"
            >
              Send
            </button>
          </div>
        </div>

        <button
          onClick={() => router.push("/collect")}
          className="bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded font-bold mt-6"
        >
          Start Over
        </button>
      </div>
    </div>
  );
} 