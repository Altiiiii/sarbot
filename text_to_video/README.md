# Metinden Video Üretici

Bu modül, sağlanan metni temel alarak her cümleyi sahneye dönüştüren basit bir yapay zekâ destekli video üretim hattı sunar. Sistem, metni sahnelere böler, her sahne için metin odaklı bir görsel üretir ve bu kareleri OpenCV yardımıyla MP4 formatında birleştirir.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OpenCV'nin MP4 çıktısı üretebilmesi için sisteminizde uygun codec'lerin kurulu olması gerekir.

## Kullanım

```bash
python -m text_to_video.main "Metninizi buraya yazın" --output video.mp4
```

Alternatif olarak metni bir dosyadan da okuyabilirsiniz:

```bash
python -m text_to_video.main --input-file senaryo.txt --output video.mp4
```

### Parametreler

- `--width` ve `--height`: Videonun çözünürlüğü.
- `--fps`: Saniyedeki kare sayısı.
- `--seconds-per-segment`: Her sahnenin süresi.
- `--font`: Özel bir `.ttf` yazı tipi dosyası kullanmak için yol belirtin.

## Nasıl Çalışır?

1. **Storyboard oluşturma:** `StoryboardBuilder` metni cümlelere böler ve her cümle için arka plan rengi seçer.
2. **Kare render etme:** `FrameRenderer`, Pillow kullanarak metni büyük puntolu olarak karelere yazar.
3. **Video derleme:** `VideoAssembler`, render edilen kareleri verilen FPS değerine göre MP4 videoya çevirir.

Bu yaklaşım, hızlı prototip oluşturmak ve metinden kısa tanıtım videoları üretmek için uygundur. Daha gelişmiş senaryolar için segment başına farklı görseller veya seslendirme modülleri eklemek üzere kodu genişletebilirsiniz.
