function preview(event)
{
	image = document.getElementById('displayImage');
	label = document.getElementById('chooseLabel');

	previewBox = document.getElementById('previewBox');
	previewBox.style.display = 'block';

	image.src = URL.createObjectURL(event.target.files[0]);
	label.innerHTML = event.target.files[0].name;
}
