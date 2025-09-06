using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

namespace Maymun_Cicegi_Tespit.Controllers
{
    public class UploadController : Controller
    {
        private readonly IHttpClientFactory _clientFactory;

        public UploadController(IHttpClientFactory clientFactory)
        {
            _clientFactory = clientFactory;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Index(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
            {
                ViewBag.Message = "Lütfen bir görsel seçin.";
                return View();
            }

            var client = _clientFactory.CreateClient();
            var requestContent = new MultipartFormDataContent();

            using (var stream = imageFile.OpenReadStream())
            {
                var imageContent = new StreamContent(stream);
                imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
                requestContent.Add(imageContent, "image", imageFile.FileName);

                var response = await client.PostAsync("http://127.0.0.1:5000/predict", requestContent);

                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var result = JsonDocument.Parse(json);
                    ViewBag.Prediction = result.RootElement.GetProperty("prediction").GetString();
                }
                else
                {
                    ViewBag.Prediction = "API'den tahmin alınamadı.";
                }
            }

            // Yüklenen resmi göstermek için dosya yolu kaydet
            var uploadsPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot/uploads");
            Directory.CreateDirectory(uploadsPath);
            var filePath = Path.Combine(uploadsPath, imageFile.FileName);
            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                await imageFile.CopyToAsync(fileStream);
            }

            ViewBag.ImagePath = "/uploads/" + imageFile.FileName;
            return View("Result");
        }
    }
}
