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
          <p>Found {searchResults.results_count} listings 
             {searchResults.stored_count !== undefined && 
              ` (${searchResults.stored_count} stored)`}
          </p>
          
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", 
            gap: "1rem" 
          }}>
            {searchResults.results.map((r: any, idx: number) => (
              <div
                key={idx}
                style={{ 
                  border: "1px solid #ccc", 
                  borderRadius: "8px",
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.5rem",
                  boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
                  transition: "transform 0.2s",
                  cursor: "pointer",
                  backgroundColor: "white"
                }}
                onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.02)")}
                onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                {r.imgSrc && (
                  <div style={{ position: "relative", paddingBottom: "66.67%" }}>
                    <img 
                      src={r.imgSrc} 
                      alt={`Listing at ${r.address}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        borderRadius: "4px"
                      }}
                      onError={(e) => {
                        // Fallback if image fails to load
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                )}
                
                <div style={{ flex: 1 }}>
                  <p style={{ fontWeight: "bold", fontSize: "1.1em", margin: "0.5rem 0" }}>
                    ${typeof r.price === 'number' ? r.price.toLocaleString() : r.price}
                  </p>
                  <p style={{ margin: "0.5rem 0", color: "#666" }}>{r.address}</p>
                  <p style={{ margin: "0.5rem 0", color: "#666" }}>
                    <span>{r.beds} beds</span>
                    {r.baths && <span> • {r.baths} baths</span>}
                    {r.sqft && <span> • {r.sqft.toLocaleString()} sqft</span>}
                  </p>
                  {r.detailUrl && (
                    <a 
                      href={r.detailUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        display: "inline-block",
                        marginTop: "0.5rem",
                        padding: "0.5rem 1rem",
                        backgroundColor: "#006aff",
                        color: "white",
                        textDecoration: "none",
                        borderRadius: "4px",
                        fontSize: "0.9em",
                        transition: "background-color 0.2s"
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0051c2")}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#006aff")}
                    >
                      View on Zillow →
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
