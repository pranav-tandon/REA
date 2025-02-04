import type { NextApiRequest, NextApiResponse } from "next";

/**
 * Forwards user queries to FastAPI's /house-search endpoint.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "POST") {
    try {
      // Determine which endpoint to use based on the request type
      const endpoint = req.body.type === "chat" ? "chat" : "house-search";
      
      const backendRes = await fetch(`http://localhost:8000/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: req.body.user_input,
          force_refresh: req.body.force_refresh,
          context: req.body.context
        }),
      });

      if (!backendRes.ok) {
        const errorData = await backendRes.json();
        console.error("Backend error:", errorData);
        return res.status(backendRes.status).json(errorData);
      }

      const data = await backendRes.json();
      return res.status(200).json(data);
    } catch (error) {
      console.error("API error:", error);
      return res.status(500).json({ error: "Backend service error" });
    }
  } else {
    return res.status(405).json({ error: "Method not allowed" });
  }
}
