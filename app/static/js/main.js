// SpotCheck — logika sisi klien.
// Diekstrak dari blok <script> spotcheck-prototype.html (baris 930-996).
// Navigasi go(), akordion, quick-nav dan scroll-spy dipertahankan apa adanya.

  function go(id){
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.getElementById('page-'+id).classList.add('active');
    document.querySelectorAll('nav.menu button').forEach(b=>b.classList.toggle('active', b.dataset.nav===id));
    document.querySelector('nav.menu').classList.remove('open');
    window.scrollTo({top:0,behavior:'instant'});
  }

  function toggleAcc(head){
    head.parentElement.classList.toggle('open');
  }

  function toggleAll(scope,btn){
    const items=document.querySelectorAll('#acc-'+scope+' .acc');
    const anyClosed=[...items].some(a=>!a.classList.contains('open'));
    items.forEach(a=>a.classList.toggle('open',anyClosed));
    btn.textContent=anyClosed?'Collapse all':'Expand all';
  }

  // Quick-nav: smooth scroll + open target + highlight
  document.querySelectorAll('.quicknav a').forEach(link=>{
    link.addEventListener('click',e=>{
      e.preventDefault();
      const target=document.querySelector(link.getAttribute('href'));
      target.classList.add('open');
      const scope=link.closest('aside').id.replace('qn-','');
      const expandBtn=document.querySelector('.expand-all[data-scope="'+scope+'"]');
      const allOpen=[...document.querySelectorAll('#acc-'+scope+' .acc')].every(a=>a.classList.contains('open'));
      if(expandBtn) expandBtn.textContent=allOpen?'Collapse all':'Expand all';
      link.closest('aside').querySelectorAll('a').forEach(a=>a.classList.remove('active'));
      link.classList.add('active');
      setTimeout(()=>target.scrollIntoView({behavior:'smooth',block:'start'}),80);
    });
  });

  // Scroll-spy for quick-nav highlighting
  const spy=new IntersectionObserver((entries)=>{
    entries.forEach(en=>{
      if(en.isIntersecting){
        const id=en.target.id;
        const link=document.querySelector('.quicknav a[href="#'+id+'"]');
        if(link){
          link.closest('aside').querySelectorAll('a').forEach(a=>a.classList.remove('active'));
          link.classList.add('active');
        }
      }
    });
  },{rootMargin:'-80px 0px -70% 0px',threshold:0});
  document.querySelectorAll('.acc').forEach(a=>spy.observe(a));

  /* ---------- Prediksi nyata pada dropzone ----------
     Menggantikan runDemo() milik prototype yang meng-hardcode 85/15. Animasi
     bar-nya dipertahankan; yang berubah hanya sumber angkanya: sekarang dari
     endpoint POST /predict. */

  const dz        = document.getElementById('dz');
  const fileInput = document.getElementById('fileInput');
  const pill      = document.getElementById('scanPill');
  const dzTitle   = document.getElementById('dzTitle');
  const dzSub     = document.getElementById('dzSub');
  const errorBox  = document.getElementById('scanError');
  const resultBox = document.getElementById('result');
  const preview   = document.getElementById('dzPreview');
  const againBtn  = document.getElementById('scanAgain');
  const cameraIn  = document.getElementById('cameraInput');
  const cameraBtn = document.getElementById('takePhoto');
  const learnBox  = document.getElementById('resultLearn');
  const learnBtn  = document.getElementById('resultLearnBtn');

  const DZ_TITLE_IDLE = dzTitle.textContent;
  const DZ_SUB_IDLE   = dzSub.innerHTML;
  const MAX_BYTES     = 8 * 1024 * 1024;   // samakan dengan MAX_CONTENT_LENGTH
  // Samakan dengan ALLOWED_EXTENSIONS di config.py. HEIC sengaja tidak ada:
  // iOS praktis selalu mengonversinya ke JPEG saat diunggah lewat form web.
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  // Prediksi biasanya selesai di bawah satu detik — terlalu cepat untuk sempat
  // membaca animasi pindainya. Tahan state "Analyzing…" selama minimal satu
  // sapuan penuh (samakan dengan durasi @keyframes scan-sweep di style.css).
  const MIN_SCAN_MS = 1200;

  let busy = false;
  let previewUrl = null;   // object URL yang sedang dipakai preview

  // Tampilkan foto terpilih di dalam dropzone.
  function showPreview(file){
    clearPreview();
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.hidden = false;
    dz.classList.add('has-photo');
  }

  // Lepaskan object URL sebelumnya supaya tidak menumpuk di memori.
  function clearPreview(){
    if(previewUrl){
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    preview.hidden = true;
    preview.removeAttribute('src');
    dz.classList.remove('has-photo');
    againBtn.hidden = true;
  }

  function showError(message){
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError(){
    errorBox.hidden = true;
    errorBox.textContent = '';
  }

  let lastName = '';

  function setBusy(on, filename){
    busy = on;
    if(filename) lastName = filename;
    dz.classList.toggle('busy', on);
    pill.textContent    = on ? 'Analyzing…' : 'Model ready';
    dzTitle.textContent = on ? 'Analyzing…' : DZ_TITLE_IDLE;

    if(on){
      dzSub.textContent = lastName;
    }else if(dz.classList.contains('has-photo')){
      // Foto masih tampil: pertahankan nama berkasnya agar jelas apa yang dianalisis.
      dzSub.textContent = lastName;
    }else{
      dzSub.innerHTML = DZ_SUB_IDLE;
    }

    // Tombol ganti foto hanya relevan setelah ada foto dan analisis selesai.
    againBtn.hidden = on || !dz.classList.contains('has-photo');
  }

  // Tunggu sisa waktu agar animasi pindai sempat terlihat utuh.
  function waitForMinimumScan(startedAt){
    const sisa = MIN_SCAN_MS - (performance.now() - startedAt);
    return sisa > 0 ? new Promise(resolve => setTimeout(resolve, sisa)) : Promise.resolve();
  }

  // Validasi di sisi klien; server tetap memvalidasi ulang.
  function validate(file){
    if(!ALLOWED_TYPES.includes(file.type)){
      return 'Please choose a JPG, PNG or WebP image.';
    }
    if(file.size > MAX_BYTES){
      return 'That image is larger than 8 MB. Please choose a smaller one.';
    }
    return null;
  }

  function renderResult(data){
    resultBox.classList.add('show');

    document.getElementById('verdict').textContent = 'Looks like ' + data.verdict;

    const chip = document.getElementById('confChip');
    chip.textContent = data.confidence + '% confidence';
    chip.className = 'chip ' + (data.verdict === 'Tinea' ? 'chip-tinea' : 'chip-eczema');

    // Antar pengguna ke halaman edukasi yang sesuai hasilnya.
    const tinea = data.verdict === 'Tinea';
    learnBtn.textContent = 'Read the ' + (tinea ? 'Tinea' : 'Eczema') + ' guide →';
    learnBtn.onclick = () => go(tinea ? 'tinea' : 'eczema');
    learnBox.hidden = false;

    // Reset ke 0 supaya animasi terputar ulang setiap prediksi (perilaku prototype).
    document.getElementById('barE').style.width = '0%';
    document.getElementById('barT').style.width = '0%';
    requestAnimationFrame(()=>{
      setTimeout(()=>{
        document.getElementById('barE').style.width = data.eczema_pct + '%';
        document.getElementById('barT').style.width = data.tinea_pct + '%';
        document.getElementById('valE').textContent = data.eczema_pct + '%';
        document.getElementById('valT').textContent = data.tinea_pct + '%';
      },80);
    });
    setTimeout(()=>resultBox.scrollIntoView({behavior:'smooth',block:'center'}),200);
  }

  async function runPrediction(file){
    if(busy) return;

    clearError();
    const problem = validate(file);
    if(problem){
      clearPreview();
      resultBox.classList.remove('show');
      showError(problem);
      return;
    }

    showPreview(file);
    setBusy(true, file.name);
    const startedAt = performance.now();
    try{
      const body = new FormData();
      body.append('image', file);

      const res  = await fetch('/predict', {method:'POST', body:body});
      const data = await res.json().catch(()=>null);

      await waitForMinimumScan(startedAt);

      if(!res.ok){
        clearPreview();
        resultBox.classList.remove('show');
        showError((data && data.error) || 'Could not analyze that image. Please try again.');
        return;
      }
      renderResult(data);
    }catch(err){
      await waitForMinimumScan(startedAt);
      clearPreview();
      resultBox.classList.remove('show');
      showError('Could not reach the server. Check your connection and try again.');
    }finally{
      setBusy(false);
    }
  }

  // Klik untuk memilih berkas.
  dz.addEventListener('click', ()=>{ if(!busy) fileInput.click(); });
  againBtn.addEventListener('click', ()=>{ if(!busy) fileInput.click(); });

  // Foto langsung. Atribut capture="environment" pada input membuat ponsel
  // membuka kamera belakang; desktop mengabaikannya (tombolnya memang
  // disembunyikan di desktop lewat @media (pointer:coarse) di style.css).
  cameraBtn.addEventListener('click', ()=>{ if(!busy) cameraIn.click(); });

  [fileInput, cameraIn].forEach(input=>{
    input.addEventListener('change', ()=>{
      if(input.files.length) runPrediction(input.files[0]);
      // Kosongkan agar memilih berkas yang sama dua kali tetap memicu 'change'.
      input.value = '';
    });
  });

  // Drag-and-drop.
  ['dragenter','dragover'].forEach(evt=>{
    dz.addEventListener(evt, e=>{
      e.preventDefault();
      if(!busy) dz.classList.add('dragover');
    });
  });
  ['dragleave','dragend'].forEach(evt=>{
    dz.addEventListener(evt, ()=>dz.classList.remove('dragover'));
  });
  dz.addEventListener('drop', e=>{
    e.preventDefault();
    dz.classList.remove('dragover');
    if(e.dataTransfer.files.length) runPrediction(e.dataTransfer.files[0]);
  });

  // Cegah browser membuka berkas bila dilepas di luar dropzone.
  ['dragover','drop'].forEach(evt=>{
    window.addEventListener(evt, e=>{
      if(!dz.contains(e.target)) e.preventDefault();
    });
  });
