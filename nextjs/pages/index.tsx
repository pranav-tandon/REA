import { useState } from "react";

export default function Home() {
  const [userInput, setUserInput] = useState("");
  const [searchResults, setSearchResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setSearchResults(null);
    try {
      const res = await fetch("/api/houseSearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userInput }),
      });
      const data = await res.json();
      setSearchResults(data);
    } catch (err) {
      console.error(err);
      alert("Error searching.");
    }
    setLoading(false);
  };

  const handleRealEstateSearch = async (userInput: string) => {
    try {
      const response = await fetch("/api/houseSearch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_message: userInput }),
      });
      const data = await response.json();
      console.log("Real Estate Search Results:", data);
    } catch (error) {
      console.error("Error during real estate search:", error);
    }
  };

  const handleBasicChat = async (userInput: string) => {
    try {
      const response = await fetch("/api/basicChat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_message: userInput }),
      });
      const data = await response.json();
      console.log("Chat Response:", data);
    } catch (error) {
      console.error("Error during chat:", error);
    }
  };

  return (
    <main style={{ padding: "1rem" }}>
      <h1>REA - House Search</h1>
      <textarea
        rows={3}
        style={{ width: "100%", marginBottom: "0.5rem" }}
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        placeholder="E.g. 'Looking to rent a 2 bedroom in Seattle for 2000'..."
      />
      <div>
        <button onClick={handleSearch} disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
      {searchResults && searchResults.results && (
        <div style={{ marginTop: "1rem" }}>
          <h3>Parsed Query:</h3>
          <pre>{JSON.stringify(searchResults.parsed_query, null, 2)}</pre>

          <h3>Results:</h3>
          <p>Found {searchResults.results_count} listings</p>
          {searchResults.results.map((r: any, idx: number) => (
            <div
              key={idx}
              style={{ border: "1px solid #ccc", margin: "1rem 0", padding: "1rem" }}
            >
              <p><strong>Address:</strong> {r.address}</p>
              <p><strong>Price:</strong> {r.price}</p>
              <p><strong>Beds:</strong> {r.beds}</p>
              <p><strong>Baths:</strong> {r.baths}</p>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
