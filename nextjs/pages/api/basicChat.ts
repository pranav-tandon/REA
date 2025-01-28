import type { NextApiRequest, NextApiResponse } from "next";

/**
 * Forwards user queries to FastAPI's /basic-chat endpoint.
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === "POST") {
    try {
      // Ensure the request body has the correct format
      const requestBody = {
        user_input: req.body.user_input
      };

      const backendRes = await fetch("http://localhost:8000/basic-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
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